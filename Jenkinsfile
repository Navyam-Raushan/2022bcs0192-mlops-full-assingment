pipeline {
    agent any

    environment {
        IMAGE_NAME = "batch2_2022bcso192"
        CONTAINER_NAME = "batch2_container"
        PORT = "8000"
    }

    stages {

        stage('Checkout') {
            steps {
                git 'https://github.com/Navyam-Raushan/2022bcs0192-mlops-full-assingment.git'
            }
        }

        stage('Setup Python') {
            steps {
                sh '''
                python3 -m venv venv
                . venv/bin/activate
                pip install --upgrade pip
                pip install -r requirements.txt
                '''
            }
        }

        stage('Train') {
            steps {
                sh '''
                . venv/bin/activate
                python train.py
                '''
            }
        }

        stage('Evaluate') {
            steps {
                sh '''
                . venv/bin/activate
                python evaluate.py
                '''
            }
        }

        stage('Docker Build') {
            steps {
                sh '''
                docker build -t $IMAGE_NAME .
                '''
            }
        }

        stage('Deploy') {
            steps {
                sh '''
                docker rm -f $CONTAINER_NAME || true
                docker run -d -p $PORT:$PORT --name $CONTAINER_NAME $IMAGE_NAME
                '''
            }
        }

        stage('Smoke Test') {
            steps {
                sh '''
                sleep 10
                curl http://localhost:$PORT/docs
                '''
            }
        }
    }

    post {
        success {
            echo "✅ SUCCESS: Model trained, deployed, API live"
        }
        failure {
            echo "❌ FAILED: Check logs in Jenkins"
        }
    }
}