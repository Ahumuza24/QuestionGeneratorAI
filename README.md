# AI Based Question Generation System

An intelligent system that generates questions from PDF and Word documents. The system can create multiple-choice questions and structured questions with answers, allowing users to modify and export them to PDF.

## Features

- **Document Upload**: Support for PDF and Word documents
- **AI-Powered Question Generation**: Generate multiple-choice and structured questions
- **Answer Generation**: Automatically generate answers for all questions
- **Question Editing**: Modify generated questions before export
- **PDF Export**: Download questions and answers as PDF
- **User Authentication**: Secure login and registration system
- **User Dashboard**: Manage uploaded documents and generated questions

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/QuestionGeneratorAI.git
cd QuestionGeneratorAI
```

2. Create a virtual environment and activate it:
```
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install the required packages:
```
pip install -r requirements.txt
```

4. Set up environment variables:
```
# Create a .env file in the root directory with the following variables
SECRET_KEY=your_secret_key
DATABASE_URI=sqlite:///questions.db
```

5. Initialize the database:
```
flask db init
flask db migrate -m "Initial migration"
flask db upgrade
```

6. Run the application:
```
flask run
```

7. Access the application at http://localhost:5000

## Usage

1. Register an account or log in
2. Upload a PDF or Word document
3. Select the type of questions to generate (multiple-choice, structured, or both)
4. Adjust generation parameters if needed
5. Generate questions
6. Review and edit questions as needed
7. Export questions and answers to PDF

## Technologies Used

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **Database**: SQLAlchemy with SQLite
- **Authentication**: Flask-Login
- **AI Models**: Transformers (Hugging Face)
- **PDF Processing**: PyPDF2, ReportLab
- **Word Processing**: python-docx

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
#