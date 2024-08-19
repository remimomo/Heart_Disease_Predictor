heart_disease_prediction/
│
├── .github/
│   └── workflows/
│       └── ci-cd-pipeline.yml  # CI/CD pipeline configuration file
│
├── app.py                      # Main Flask application
├── model.py                    # Model training, evaluation, and prediction logic
├── requirements.txt            # Dependencies list
├── templates/                  # HTML templates for the Flask app
│   └── index.html              # UI for the user to interact with the app
├── static/                     # Static files (CSS, images, etc.)
│   └── style.css               # Styling for the HTML template
├── data/                       # Directory for storing the dataset
│   └── heart-disease.csv       # Heart disease dataset
├── models/                     # Directory to store the trained model
│   └── model.pkl               # Serialized trained model
├── tests/                      # Unit tests for the application
│   └── test_app.py             # Test cases for the Flask app
└── README.md                   # Project documentation


########## Heart Disease Prediction App ##############

#### Description
This project is a Flask web application that predicts the likelihood of heart disease based on user input of medical attributes.


#### Installation

#1. Clone the repository:
git clone <repository_url> cd heart_disease_prediction
#2. Install the required Python packages:
pip install -r requirements.txt
#3. Ensure you have the dataset in the `data/` folder.

#4. Train the model and save it:
python model.py
#5. Run the Flask app locally:
python app.py
#### Usage

#1. Open the application in your web browser at `http://127.0.0.1:5000/`.
#2. Enter the required medical attributes and click "Predict" to see the #result.

#### Testing

To run the unit tests:
python -m unittest discover tests
#### Conclusion

This project demonstrates best practices in structuring a machine learning project, creating a web interface, and ensuring code quality with unit tests.
########################## Deploying with Heroku #########################
#Deploying your Flask application on Heroku is a straightforward process. Below are the #detailed steps to deploy your heart_disease_prediction project on Heroku.
######## Step 1: Set Up Heroku CLI
### Install the Heroku CLI if you haven't already. You can install it from the Heroku CLI page.
brew tap heroku/brew && brew install heroku  # For macOS
#Alternatively, you can download it directly for Windows and Linux from the Heroku #documentation.
### Log in to your Heroku account from the terminal.
heroku login
####### Step 2: Prepare Your Flask Application for Heroku Deployment
### Procfile: Create a Procfile in the root directory of your project. This file tells ### Heroku how to run your application.
# Create a file named Procfile with the following content:
web: gunicorn app:app
#This tells Heroku to use gunicorn to run your Flask application, and that the entry #point of your application is in the app.py file (app refers to the Flask instance).
#### requirements.txt: Make sure your requirements.txt file lists all the necessary Python packages, including gunicorn (which is used to serve the Flask app).
# Add gunicorn to requirements.txt if it's not already there:
Flask
numpy
pandas
scikit-learn
matplotlib
seaborn
lightgbm
gunicorn
#### runtime.txt: Specify the Python version that your application requires by ####creating a runtime.txt file in your root directory. For example:
python-3.8.12
#This tells Heroku which version of Python to use.
#### Ensure .gitignore: Make sure you have a .gitignore file to exclude ####unnecessary files from being pushed to Heroku. It might include:
__pycache__/
*.pyc
.DS_Store
env/
venv/
.venv/
####### Step 3: Create a New Heroku App
#### Create the Heroku app:
heroku create your-app-name
# Replace your-app-name with your desired app name. Heroku will automatically #assign a unique name if you don't specify one.
#### Add the Heroku remote to your git repository:
git remote add heroku https://git.heroku.com/your-app-name.git
Replace your-app-name with your actual app name.
####### Step 4: Deploy the Application
#### Commit your changes to your Git repository:
git add .
git commit -m "Prepare app for Heroku deployment"
#### Push the code to Heroku:
git push heroku main
#This command deploys your application to Heroku. The first time you do this, it may #take a few minutes as Heroku installs all dependencies and sets up your application.
####### Step 5: Manage Environment Variables
#If your application uses any environment variables (e.g., API keys, secret keys), you can set #them in Heroku.
#### Set environment variables using the Heroku CLI:
heroku config:set SECRET_KEY=your-secret-key
#Replace SECRET_KEY and your-secret-key with your actual environment variable #and value.
####### Step 6: Access Your Deployed Application
#Once the deployment is complete, you can access your application using the URL provided #by Heroku:
heroku open
# This command will open your application in your default web browser.
####### Step 7: Monitor Your Application
#Heroku provides a dashboard where you can monitor your application's performance, view #logs, and manage resources. You can access it by visiting dashboard.heroku.com and #selecting your app.
# You can also view logs directly from the terminal:
heroku logs --tail
####### Step 8: Automatic Deployments (Optional)
#If you want Heroku to automatically deploy your app whenever you push changes to the #main branch in GitHub, you can set up automatic deployments:
1.	# Go to your app's dashboard on Heroku.
2.	# Under the "Deploy" tab, connect your GitHub repository.
3.	# Enable "Automatic Deploys" from the main branch.
