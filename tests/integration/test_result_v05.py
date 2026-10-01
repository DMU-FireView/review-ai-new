from fastapi.testclient import TestClient
from app.main import app


def test_api_model_to_rti_to_flat_contract(monkeypatch):
    monkeypatch.setattr('app.services.analysis.predict_text_score', lambda content: {
        'text_score': 87, 'suspicious_probability': .13, 'predicted_label': 'NORMAL',
    })
    with TestClient(app) as client:
        response = client.post('/analysis/crawler-reviews', json={'reviews': [{
            'platform': 'kurly', 'product_id': 'p', 'review_id': 'r',
            'content': 'A sufficiently long review about my own experience using this product.',
        }]})
    assert response.status_code == 200
    assert response.json() == {
        'platform': 'kurly', 'product_id': 'p', 'review_count': 1,
        'results': [{'review_id': 'r', 'rti': 87., 'level': 'safe',
                     'text_score': 87., 'behavior_score': -1., 'network_score': -1., 'reasons': []}],
    }


def test_api_all_unavailable_never_assigns_level(monkeypatch):
    monkeypatch.setattr('app.services.analysis.predict_text_score', lambda content: {
        'text_score': -1, 'suspicious_probability': None, 'predicted_label': None,
    })
    with TestClient(app) as client:
        response = client.post('/analysis/crawler-reviews', json={'reviews': [{
            'platform': 'kurly', 'product_id': 'p', 'review_id': 'r', 'content': '',
        }]})
    assert response.status_code == 200
    row = response.json()['results'][0]
    assert row['level'] is None
    assert all(row[key] == -1 for key in ('rti', 'text_score', 'behavior_score', 'network_score'))
    assert row['reasons'] == []
