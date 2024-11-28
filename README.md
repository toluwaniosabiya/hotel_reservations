# Hotel Reservations Analysis and Pipelines: 
## *A guide on using Metaflow for creating and monitoring data pipelines and MLflow for registering and versioning models*

This project is designed to demonstrate how to build a data pipeline with Metaflow and use MLflow to register and version any trained model in the MLflow model registry. The dataset used is that of a hotel reservation company who want to know given some search parameters the hotel a user will likely book and the probability that the user will book any particular hotel. 

## Features

- **Data Pipeline Automation**: Includes modular pipelines for preprocessing, feature engineering, and model training.
- **Exploratory Data Analysis**: Contains a Jupyter Notebook (`hotel-reservations.ipynb`) for initial exploration and analysis of the dataset as well as step-by-step demonstration of the code that eventually goes into the pipeline.
- **Dependency Management**: Uses `requirements.in` and `requirements.txt` for environment setup.

## Directory Structure

```plaintext
hotel_reservations-main/
├── data/                              # Contains datasets
├── pipelines/                         # Includes pipeline-related scripts
├── LICENSE                            # Project licensing details
├── README.md                          # Project documentation (you are reading this)
├── requirements.in                    # Input dependency specifications
├── requirements.txt                   # Locked dependencies for environment setup
├── hotel-reservations.ipynb           # Jupyter Notebook for exploration
```

## Exploration Instructions

1. Clone the repository into a folder of your choice or fork to have your own copy:
   ```bash
   git clone https://github.com/toluwaniosabiya/hotel_reservations.git
   ```

2. Navigate into the cloned directory, `hotel_reservations` with an editor of your choice (assuming your directory is hotel_reservations)
   ```bash
   cd hotel_reservations
   ```

3. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scriptsctivate`
   ```
   **Note:** You might need to install pyenv to create a virtual environment. For more information on installing pyenv, see (https://github.com/pyenv/pyenv-installer). 

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Configure MLflow tracking uniform resource identifier, pointing it to a port on your local device as follows: 
   ```bash
   export $((echo "MLFLOW_TRACKING_URI=http://127.0.0.1:8080" >> .env; cat .env) | xargs)
   ```
   This will set MLFLOW_TRACKING_URI as "http://127.0.0.1:8080" in your .env file.

6. By default, MLflow tracks experiments and stores data in files inside a local `./mlruns` directory. You can change the location of the tracking directory or use a SQLite database using the parameter `--backend-store-uri`. The following example uses a SQLite database to store the tracking data:

   ```bash
   mlflow server --host 127.0.0.1 --port 8080 \
      --backend-store-uri sqlite:///hotel-reservations.db
   ```
   This means that the mlflow server will run on your localhost on port 8080 and the backend store will be set to hotel-reservations.db. Since this is a sqlite database, the file will be created if it doesn't exist and if it does, tracking information will be stored in that database.


## Execute Pipelines
Run the `training.py` file in the `pipelines` directory follows:
```bash
python pipelines/training.py run
```

## View Results
Because the data pipeline is built using MLflow, you can visualise the pipeline and get more information about it by running the following comand in the terminal:
```bash
python pipelines/training.py card server
```
By default, Metaflow UI operates on port 8324 on your local device. So, navigate to localhost:8324 on your web browser to see the Metaflow UI.

## Visualize Registered Model
Run the following in the terminal to open up the MLflow user interface:
```bash
mlflow ui --port 8081 --backend-store-uri sqlite:///hotel-reservations.db
```
**Note:** Remember to use different a different port to the one hosting the MLflow server to avoid conflicts

Open your web browser and nagivate to your localhost on port 8081 as created earlier (http://127.0.0.1:8081). The MLflow UI will open up and you can see the runs and models that have been registered, with all the information about the models.

## Contributing

1. Fork the repository.
2. Create a new feature branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your message"
   ```
4. Push the branch and submit a pull request.

## License

This project is licensed under the terms of the [LICENSE](./LICENSE) file.

## Contact

For questions or feedback, please open an issue or reach out to the maintainer.
