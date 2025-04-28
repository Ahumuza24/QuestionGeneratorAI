from app import create_app, db

app = create_app()

with app.app_context():
    # This will create all tables from scratch
    db.create_all()

print("Database tables created successfully!")