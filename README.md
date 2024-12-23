# Hotel Reservations Analysis and Pipelines: 
## *A guide on using Metaflow for creating and monitoring data pipelines and MLflow for registering and versioning models*

This project is designed to demonstrate how to build a data pipeline with Metaflow and use MLflow to register and version any trained model in the MLflow model registry. The dataset used is that of a hotel reservation company who want to given some search parameters predict the hotel a user will likely book and the probability of that booking happening. 

## Features

- **Data Pipeline Automation**: Includes modular pipelines for preprocessing, feature engineering, model training, model registration, and inference.
- **Dependency Management**: Uses `requirements.txt` for environment setup.

## Directory Structure

```plaintext
hotel_reservations-main/
├── data/                              # Contains datasets
├── pipelines/                         # Includes pipeline-related scripts
├── LICENSE                            # Project licensing details
├── output.json                        # Example json file for running inference
├── README.md                          # Project documentation (you are reading this)
├── requirements.in                    # Input dependency specifications
├── requirements.txt                   # Locked dependencies for environment setup
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

6. By default, MLflow tracks experiments and stores data in files inside a local `./mlruns` directory. You can change the location of the tracking directory or use a SQLite database using the flag `--backend-store-uri`. Run the following example in a new terminal window to use a SQLite database to store the tracking data:

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
Because the data pipeline is built using Metaflow, you can visualise the pipeline and get more information about it by running the following comand in a new terminal window:
```bash
python pipelines/training.py card server
```
By default, Metaflow UI operates on port 8324 on your local device. So, navigate to localhost:8324 on your web browser to see the Metaflow UI.

## Visualize Registered Model
Run the following in a new terminal window to open up the MLflow user interface:
```bash
mlflow ui --port 8081 --backend-store-uri sqlite:///hotel-reservations.db
```
**Note:** Remember to use different a different port to the one hosting the MLflow server to avoid conflicts

Open your web browser and nagivate to your localhost on port 8081 as created earlier (http://127.0.0.1:8081). The MLflow UI will open up and you can see the runs and models that have been registered, with all the information about the models.

## Running inference from saved model locally
You can serve the trained model locally leveraging MLflow's inbuilt flask server by running the following general syntax in a new terminal window:
```bash
mlflow models serve -m models:/model-name/version -p 8082 --no-conda # You can omit this flag if your environment is set up with conda 
```
In this case, that would be 
```bash
mlflow models serve -m models:/hotel-reservations/latest -p 8082 --no-conda 
```
**Note:** You should also use a different port for this. Also, latest above means the latest version of the model with name 'hotel-reservations'. You can hard-code the version number if you wish to do so.

To have the trained model serve up predictions locally, you can  to pass in your inputs as json or csv and get predictions. There are two ways to go about this.
1. If you have a json file with the following general format:
   ```json
   {"inputs": [
      {"key": "value", ...},
      {"key": "value", ...},
      ...
   ]}
   ```
   Then you can run inferent using the following bash script
   ```bash
   curl -X POST http://127.0.0.1:8082/invocations \
   -H 'Content-Type: application/json' \
   --data-binary @output.json
   ```
   Where output.json is your json file. See the output.json file in the repo for your reference.

   **Note:** The json file must start with the key 'inputs' and the value will be a list of parameters, in this case a list of dictionaries/json objects with key-value pairs corresponding to the different variables/columns in the original data and their values.

2. You can pass in the contents of the json file directly as follows:
   ```bash
   curl -X POST http://127.0.0.1:8082/invocations \
   -H 'Content-Type: application/json' \
   -d '{"inputs": [
      {"key": "value", ...},
      {"key": "value", ...},
      ...
   ]}'
   ```

## Summary
Metaflow makes it easy to build data pipelines and MLflow makes it possible to version models and serve them easily, eliminating the need for manual tracking.

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
