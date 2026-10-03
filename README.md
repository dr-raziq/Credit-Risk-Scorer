Credit Risk Scorer

    An explainable machine learning system for credit risk assessment using the Lending Club dataset.


What It Does

    •	Predicts the probability of loan default (charged off) for a given loan application.
    •	Converts the default probability into a credit score between 300 and 850.
    •	Assigns a risk band and an automated decision (Approve, Manual Review, Decline).
    •	Provides local explanations (per applicant) and global feature importance using SHAP.
    •	Exposes a REST API and a minimal web interface for demonstration.


Why

    This project is designed to align with European regulatory requirements for automated credit scoring:

        •	GDPR Article 15(1)(h) - grants data subjects the right to meaningful information about the logic involved in automated decisions, including credit profiling.
        •	EBA Guidelines on Loan Origination and Monitoring (EBA/GL/2020/06) - require credit institutions to document and explain creditworthiness assessments, including the ability to explain acceptance or denial.
        •	EU AI Act - and the revised Consumer Credit Directive further reinforce transparency and explainability obligations for credit scoring systems.

    SHAP (SHapley Additive exPlanations) provides both global and local explanations that satisfy these transparency requirements. The project does not express any personal or political views.


Stack

    •	Python 3.11
    •	scikit-learn, XGBoost
    •	SHAP
    •	FastAPI, Uvicorn
    •	HTML/CSS/JavaScript
    •	pytest
    •	Docker
    •	Render, Railway, or Vercel (free tiers)


How to Run

    1. Clone the repository
    git clone https://github.com/yourusername/credit-risk-scorer.git
    cd credit-risk-scorer
    2. Install dependencies
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    3. Download the dataset
    Download the Lending Club dataset from Kaggle:
    https://www.kaggle.com/datasets/wordsforthewise/lending-club
    Place accepted_2007_to_2018Q4.csv in data/ and rename it to loan.csv.
    4. Train the model
    python -m src.train_model
    This saves the trained model and preprocessor to the models/ directory.
    5. Run the API locally
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
    The API will be available at http://localhost:8000. Interactive docs at /docs.
    6. Serve the frontend
    Open frontend/index.html in a browser, or serve it with any static file server. Update the API_URL constant in the script if your API is running on a different host.
    7. Run tests
    pytest tests/ -v


Deployment

    Render (recommended)
        1.	Push the repository to GitHub.
        2.	Create a new Web Service on Render.
        3.	Connect your GitHub repository.
        4.	Render will detect Python and use the render.yaml configuration.
        5.	Set the environment variable PYTHON_VERSION to 3.11.0 if not already set.
        6.	Deploy.
    Note: Free tier services on Render sleep after 15 minutes of inactivity. The first request after a cold start may take 30–60 seconds.

    Railway
        1.	Connect your GitHub repository to Railway.
        2.	Railway will auto-detect Python.
        3.	Add the environment variable PYTHON_VERSION set to 3.11.0.
        4.	Set the start command to uvicorn api.main:app --host 0.0.0.0 --port $PORT.
        5.	Deploy.
    Railway builds are typically faster than Render.

    Vercel
    Vercel supports FastAPI through its Python runtime. Create an api/index.py entrypoint that imports the FastAPI app, and a vercel.json configuration file. Note that Vercel has a 10-second function timeout on free tiers, which may be insufficient for cold starts with SHAP.


Structure

    Credit Risk Scorer
    |_ data                
    |_ src
    |	|_ __init__.py
    |	|_ data_loader.py          
    |	|_ feature_engineer.py     
    |	|_ train_model.py          
    |	|_ explain.py              
    |	|_ scoring.py
    |              
    |_ api
    |	|_ __init__.py
    |	|_ main.py                 
    |	|_ schemas.py
    |              
    |_ frontend
    |	|_ index.html  
    |            
    |_ tests
    |	|_ __init__.py
    |	|_ test_data_loader.py
    |	|_ test_feature_engineer.py
    |	|_ test_scoring.py
    |	|_ test_api.py
    |
    |_ models                     
    |_ notebooks
    |	|_ exploration.ipynb
    |
    |_ requirements.txt
    |_ Dockerfile
    |_ render.yaml
    |_ README.md
    |_ .gitignore
    |_ LICENSE
