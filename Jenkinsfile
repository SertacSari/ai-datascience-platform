pipeline {
    agent any

    stages {
        stage('Repository Hygiene') {
            steps {
                sh '''
                    forbidden_files="$(
                        git ls-files -- .env backend/.env venv backend/uploads frontend/dist frontend/node_modules |
                        grep -vFx 'backend/uploads/.gitkeep' || true
                    )"

                    if [ -n "$forbidden_files" ]; then
                        echo "ERROR: Dangerous generated, secret, or uploaded files are tracked:"
                        echo "$forbidden_files"
                        exit 1
                    fi

                    echo "Repository hygiene check passed."
                '''
            }
        }
        stage('Backend Tests') {
            steps {
                sh '''
                    "${PYTHON_BIN:-python3}" -m venv venv
                    ./venv/bin/python -m pip install -r backend/requirements-dev.txt
                    PYTHONDONTWRITEBYTECODE=1 ./venv/bin/python -m pytest backend/tests -q
                '''
            }
        }
        stage('Frontend Build') {
            steps {
                dir('frontend') {
                    sh 'npm ci'
                    sh 'npm run build'
                }
            }
        }
    }
}
