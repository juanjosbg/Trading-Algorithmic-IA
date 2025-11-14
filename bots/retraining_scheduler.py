import schedule
import time
import subprocess
from datetime import datetime

def retrain():
    print("\n🔄 Ejecutando re-entrenamiento automátido...")
    result = subprocess.run(["python", "ml_model.py"])

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if result.returncode == 0:
        print(f"✅ {timestamp} : Entrenamiento completado.")
    else:
        print(f"❌ {timestamp} : Error entrenando modelos.")

def main():
    print("📅 Re-entrenamiento diario activado.")
    schedule.every().day.at("18:00").do(retrain)

    while True:
        schedule.run_pending()
        time.sleep(10)

if __name__ == "__main__":
    main()
