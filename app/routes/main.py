from flask import Blueprint, render_template, redirect, url_for
from flask_login import login_required, current_user
from datetime import datetime 

main = Blueprint('main', __name__)

@main.route('/')
def index():
    """Home page route"""
    return render_template('index.html')

@main.route('/dashboard')
@login_required
def dashboard():
    """User dashboard route"""
    return render_template('dashboard.html', user=current_user, now=datetime.utcnow())

@main.route('/about')
def about():
    """About page route"""
    return render_template('about.html')

@main.route('/contact')
def contact():
    """Contact page route"""
    return render_template('contact.html') 