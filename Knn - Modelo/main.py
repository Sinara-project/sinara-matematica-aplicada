# ======================================
# Conexão com Flask
# ======================================

from flask import Flask
from KnnClass import agua_bp
import os
from dotenv import load_dotenv

# Carrega variáveis do .env
load_dotenv()

app = Flask(__name__)

# Registra o módulo de previsão da água
app.register_blueprint(agua_bp, url_prefix="/agua")

@app.route("/")
def home():
    return "API Flask ativa"

if __name__ == "__main__":
    app.run(
        host=os.getenv("FLASK_HOST", "0.0.0.0"),
        port=int(os.getenv("FLASK_PORT", 5000))
    )
