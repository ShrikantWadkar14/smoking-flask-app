# Health Signal Analytics: Smoker Status Classification

This project implements a machine learning-based system to classify smoker status using health signal data. The application provides a web interface for users to input health metrics and receive predictions about smoking status.

## 🚀 Features

- Web-based interface for easy interaction
- Machine learning model for smoker status classification
- Real-time predictions
- User-friendly input form for health metrics
- Responsive design

## 📋 Prerequisites

Before running this project, make sure you have the following installed:
- Python 3.7 or higher
- pip (Python package installer)

## 🛠️ Installation

1. Clone the repository:
```bash

```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Running the Application

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

## 📁 Project Structure

```
health-signal-analytics/
├── app.py              # Main Flask application
├── requirements.txt    # Project dependencies
├── model.pkl          # Trained machine learning model
├── Data/              # Dataset directory
├── Demo_Model/        # Model demonstration files
├── Notebook/          # Jupyter notebooks for analysis
├── static/            # Static files (CSS, JS, images)
└── templates/         # HTML templates
```

## 🧪 Model Details

The project uses a LightGBM classifier trained on health signal data to predict smoker status. The model is saved in `model.pkl` and can be loaded for predictions.

## 📊 Dependencies

- Flask: Web framework
- numpy: Numerical computing
- pandas: Data manipulation
- scikit-learn: Machine learning utilities
- lightgbm: Gradient boosting framework
- joblib: Model persistence

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## Developed by 
Shrikant Wadkar
📧 shrikantwadkar100@gmail.com
🔗 https://github.com/ShrikantWadkar14

## Snapshots
![Screenshot 2025-06-18 164724](https://github.com/user-attachments/assets/5a50b043-63f9-4faa-93c8-918e0f7593da)
![Screenshot 2025-06-18 164803](https://github.com/user-attachments/assets/b109cc37-f020-4035-8179-43137eae251c)
![Screenshot 2025-06-18 165034](https://github.com/user-attachments/assets/248ea595-e8e9-410b-bc68-5c134aa74a06)
![Screenshot 2025-06-18 165101](https://github.com/user-attachments/assets/19f5715a-e4da-4b0a-9c0a-167139084faf)
![Screenshot 2025-06-18 165139](https://github.com/user-attachments/assets/078a4cb1-1369-4534-8bea-ad9cf4644a37)
![Screenshot 2025-06-18 165214](https://github.com/user-attachments/assets/1011a8f3-0fad-4c99-83dc-1b203bb412e3)




