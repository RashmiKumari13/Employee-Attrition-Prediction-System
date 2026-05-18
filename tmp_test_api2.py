from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd() / "backend"))
import app

client = app.app.test_client()
print('root', client.get('/').status_code)
print('prefixed root', client.get('/Employee-Attrition-Prediction-System/').status_code)
print('prefixed index', client.get('/Employee-Attrition-Prediction-System/index.html').status_code)
