from unittest.mock import MagicMock, patch
import json
import time
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.infrastructure.persistence.postgres_feedback import FeedbackStore
from src.api.routers import feedback
from src.api import state

def setup_function():
    state.feedback_log.clear()

def test_feedback_store_insert():
    mock_pool = MagicMock()
    mock_conn = MagicMock()
    mock_pool.connection.return_value.__enter__.return_value = mock_conn
    
    store = FeedbackStore(mock_pool)
    entry = {
        "id": "123",
        "ts": 1600000000.0,
        "conversation_id": "c_123",
        "query": "hello",
        "rating": -1,
        "comment": "bad",
        "tags": ["test"],
        "trace_id": "t_456"
    }
    
    store.insert(entry)
    
    mock_conn.execute.assert_called_once()
    args, kwargs = mock_conn.execute.call_args
    sql = args[0]
    params = args[1]
    
    assert "INSERT INTO feedback" in sql
    assert params[0] == "123"
    assert params[1] == datetime.fromtimestamp(1600000000.0, tz=timezone.utc)
    assert params[2] == "c_123"
    assert params[3] == "hello"
    assert params[4] == -1
    assert params[5] == "bad"
    assert params[6] == '["test"]'
    assert params[7] == "t_456"
    mock_conn.commit.assert_called_once()


def test_feedback_store_insert_swallows_db_error():
    mock_pool = MagicMock()
    mock_conn = MagicMock()
    mock_pool.connection.return_value.__enter__.return_value = mock_conn
    mock_conn.execute.side_effect = Exception("DB error")
    
    store = FeedbackStore(mock_pool)
    entry = {"id": "1", "ts": 1.0, "query": "q", "rating": 1}
    
    store.insert(entry)  # should not raise


def test_feedback_store_get_negative():
    mock_pool = MagicMock()
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    
    mock_pool.connection.return_value.__enter__.return_value = mock_conn
    mock_conn.execute.return_value = mock_cur
    
    # id, ts, conversation_id, query, rating, comment, tags, trace_id
    mock_ts = datetime.fromtimestamp(1600000000.0, tz=timezone.utc)
    mock_cur.fetchall.return_value = [
        ("123", mock_ts, "c_1", "q1", -1, "c1", '["t1"]', "tr1"),
    ]
    
    store = FeedbackStore(mock_pool)
    results = store.get_negative(limit=10)
    
    assert len(results) == 1
    res = results[0]
    assert res["id"] == "123"
    assert res["query"] == "q1"
    assert res["rating"] == -1
    assert res["comment"] == "c1"
    assert res["tags"] == ["t1"]
    assert res["trace_id"] == "tr1"
    assert res["ts"] == 1600000000.0

@patch("src.api.routers.feedback.get_pool")
@patch("src.api.routers.feedback.FeedbackStore")
def test_feedback_router_persists_to_db(mock_store_class, mock_get_pool):
    mock_store = mock_store_class.return_value
    
    app = FastAPI()
    app.include_router(feedback.router)
    client = TestClient(app)
    
    response = client.post(
        "/api/feedback",
        json={
            "query": "hello",
            "rating": -1,
            "comment": "bad",
            "tags": ["demo"],
            "trace_id": "tr_1"
        }
    )
    
    assert response.status_code == 200
    assert response.json()["ok"] is True
    
    mock_store.insert.assert_called_once()
    called_entry = mock_store.insert.call_args[0][0]
    assert called_entry["query"] == "hello"
    assert called_entry["trace_id"] == "tr_1"


@patch("src.api.routers.feedback.get_pool")
def test_feedback_router_no_db_fallback(mock_get_pool):
    mock_get_pool.side_effect = RuntimeError("No pool")
    
    app = FastAPI()
    app.include_router(feedback.router)
    client = TestClient(app)
    
    response = client.post(
        "/api/feedback",
        json={
            "query": "fallback test",
            "rating": 1,
        }
    )
    
    assert response.status_code == 200
    assert response.json()["ok"] is True
    
    assert len(state.feedback_log) == 1
    assert state.feedback_log[0]["query"] == "fallback test"


@patch("src.api.routers.feedback.get_pool")
@patch("src.api.routers.feedback.FeedbackStore")
def test_negative_endpoint_returns_db_results(mock_store_class, mock_get_pool):
    mock_store = mock_store_class.return_value
    mock_store.get_negative.return_value = [
        {"id": "db_1", "query": "db_q", "rating": -1}
    ]
    
    app = FastAPI()
    app.include_router(feedback.router)
    client = TestClient(app)
    
    response = client.get("/api/feedback/negative")
    assert response.status_code == 200
    
    data = response.json()
    assert len(data) == 1
    assert data[0]["id"] == "db_1"
