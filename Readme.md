## File Structure
- app
    - model
        - model.py (model class: functions to interact with the model)
        - trained_pipeline-0.1.0.pkl (pretrained model pickle file)
    - main.py (main file to run the FastAPI and api endpoints)
- .gitignore
- Readme.md

## How to run the project
```bash
   pip install fastapi uvicorn
```
```bash
   pip install -r requirements.txt
```
```bash
   uvicorn app.main:app --reload
```

## Model
[Trained Pipeline - Colab File](https://colab.research.google.com/drive/1fiUL3ff1wI3YaGa9Btyu5cEY_ifX9oZ8?usp=sharing)


