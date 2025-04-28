import torch
import re
import random
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, pipeline
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Check for CUDA availability
device = 0 if torch.cuda.is_available() else -1

# Initialize Gemini validation if available
try:
    from app.utils.gemini_validator import test_gemini_connection, batch_validate_questions
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    logger.warning("Gemini validation not available. Using T5 as fallback.")

def check_gemini_availability():
    """Check if Gemini API is available by making a small test request - no caching"""
    try:
        from app.utils.gemini_validator import test_gemini_connection
        gemini_available = test_gemini_connection()
        logger.info(f"Gemini API availability check: {'Available' if gemini_available else 'Unavailable'}")
        return gemini_available
    except Exception as e:
        logger.error(f"Gemini API is not available: {e}")
        return False

def generate_questions(context, num_questions=5, question_types=None, difficulty=None, use_gemini=True):
    """
    Generate questions using the best available model
    
    Args:
        context: Document content to generate questions from
        num_questions: Number of questions to generate
        question_types: Types of questions (multiple_choice, structured, or both)
        difficulty: Question difficulty level
        use_gemini: Whether to try using Gemini first (kept for backwards compatibility)
        
    Returns:
        List of generated question dictionaries
    """
    # Always try to use the best available model without logging model names
    logger.info(f"Generating {num_questions} questions with types: {question_types}, difficulty: {difficulty}")
    
    # Normalize question types
    if question_types is None or question_types == "both":
        question_types = ["multiple_choice", "structured"]
    elif isinstance(question_types, str):
        question_types = [question_types]
    
    # Handle different question type naming conventions
    normalized_types = []
    for qt in question_types:
        if qt == "structured":
            normalized_types.append("short_answer")
        else:
            normalized_types.append(qt)
    question_types = normalized_types
    
    # Process document text into chunks
    context_chunks = split_document_into_chunks(context)
    if not context_chunks:
        logger.warning("No valid context chunks extracted from document")
        return []
    
    # Silently check if Gemini is available
    gemini_available = False
    try:
        from app.utils.gemini_validator import test_gemini_connection
        gemini_available = test_gemini_connection()
    except Exception:
        # Silently handle the exception
        pass
    
    # Try primary model first
    questions = []
    if gemini_available:
        try:
            from app.utils.gemini_validator import generate_questions_with_gemini
            questions = generate_questions_with_gemini(context, num_questions, question_types, difficulty)
            logger.info(f"Generated {len(questions)} questions with primary model")
        except Exception:
            # Silently handle the exception
            questions = []
    
    # If primary model failed or wasn't available, use fallback
    if not questions:
        logger.info(f"Using fallback model for question generation")
        questions = generate_questions_with_t5(context_chunks, num_questions, question_types, difficulty)
        logger.info(f"Generated {len(questions)} questions with fallback model")
    
    # If all else fails, use template-based questions as a last resort
    if not questions:
        logger.warning("Using template-based questions")
        questions = generate_template_questions(context_chunks, num_questions)
        logger.info(f"Generated {len(questions)} template-based questions")
    
    # Ensure we have the requested number of questions
    if len(questions) > num_questions:
        questions = questions[:num_questions]
    
    logger.info(f"Returning {len(questions)} questions")
    return questions

def split_document_into_chunks(text, chunk_size=400, overlap=100):
    """Split document text into chunks for processing - reduced chunk size for T5"""
    if not isinstance(text, str) or len(text.strip()) < 50:
        return [text] if text else []
    
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk)
            # Start new chunk with overlap
            overlap_start = max(0, len(current_chunk) - overlap)
            current_chunk = current_chunk[overlap_start:] + " " + sentence
        else:
            current_chunk += (" " if current_chunk else "") + sentence
    
    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def generate_questions_with_t5(context_chunks, num_questions=5, question_types=None, difficulty=None):
    """Generate questions using T5 model - simplified version"""
    if question_types is None:
        question_types = ["multiple_choice", "short_answer"]
    
    logger.info(f"Initializing T5 models for question generation")
    
    # Handle potential model loading issues
    try:
        # Load tokenizer with reduced max length
        tokenizer = T5Tokenizer.from_pretrained(
            'valhalla/t5-base-e2e-qg',
            model_max_length=384  # Reduced from 512
        )
        model = AutoModelForSeq2SeqLM.from_pretrained('valhalla/t5-base-e2e-qg')
        
        # Initialize the pipelines
        question_generator = pipeline(
            'text2text-generation',
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        
        answer_generator = pipeline(
            'question-answering', 
            model='deepset/roberta-base-squad2', 
            device=device
        )
    except Exception as e:
        logger.error(f"Failed to initialize T5 models: {e}")
        return []
    
    # Process chunks sequentially to avoid threading issues
    candidate_questions = []
    
    # Use smaller chunks to prevent token overflow
    smaller_chunks = []
    for chunk in context_chunks:
        # Further split large chunks
        if len(chunk) > 300:
            sentences = sent_tokenize(chunk)
            current_small_chunk = ""
            
            for sentence in sentences:
                if len(current_small_chunk) + len(sentence) > 300:
                    if current_small_chunk:
                        smaller_chunks.append(current_small_chunk)
                    current_small_chunk = sentence
                else:
                    current_small_chunk += (" " if current_small_chunk else "") + sentence
            
            if current_small_chunk:
                smaller_chunks.append(current_small_chunk)
        else:
            smaller_chunks.append(chunk)
    
    logger.info(f"Processing {len(smaller_chunks)} small chunks")
    
    # Process each chunk sequentially
    for i, chunk in enumerate(smaller_chunks):
        if i >= 10:  # Limit number of chunks processed to avoid excessive processing time
            break
            
        logger.info(f"Processing chunk {i+1}/{len(smaller_chunks)}: {len(chunk)} chars")
        
        try:
            # Skip very short chunks
            if len(chunk) < 50:
                continue
                
            # Ensure chunk isn't too long for the model
            if len(chunk) > 300:
                chunk = chunk[:300]
            
            # Create simple input for question generation
            input_text = f"generate question: {chunk}"
            
            # Generate questions with conservative parameters
            try:
                output = question_generator(
                    input_text,
                    max_length=64,
                    num_return_sequences=1,  # Just one question per chunk for reliability
                    num_beams=2,  # Reduced beam search
                    temperature=0.7,
                    do_sample=True
                )
            except Exception as e:
                logger.error(f"Error in question generation for chunk {i+1}: {e}")
                continue
            
            # Process each generated question
            for item in output:
                question_text = item['generated_text'].strip()
                
                # Ensure it's a valid question
                if question_text and "?" in question_text:
                    # Generate answer if possible
                    try:
                        answer_output = answer_generator(
                            question=question_text,
                            context=chunk,
                            max_answer_len=30
                        )
                        
                        answer_text = answer_output.get('answer', '')
                        
                        # Skip if answer is too short
                        if len(answer_text) < 2:
                            continue
                        
                        # Add to questions list
                        candidate_questions.append({
                            "question": question_text,
                            "answer": answer_text,
                            "context": chunk,
                            "source": "t5"
                        })
                        
                        logger.info(f"Generated question: {question_text}")
                        
                        # If we have enough questions, stop
                        if len(candidate_questions) >= num_questions * 2:
                            break
                            
                    except Exception as e:
                        logger.error(f"Error generating answer: {e}")
                        
        except Exception as e:
            logger.error(f"Error processing chunk {i+1}: {e}")
    
    # Filter questions for quality
    filtered_questions = [q for q in candidate_questions if is_good_question(q["question"])]
    logger.info(f"Generated {len(filtered_questions)} quality questions after filtering")
    
    # If we still don't have any questions, generate some simple questions
    if not filtered_questions:
        logger.warning("No questions generated with T5, falling back to template-based questions")
        filtered_questions = generate_template_questions(context_chunks[:2], num_questions)
    
    # Assign question types
    for i, question in enumerate(filtered_questions):
        q_type = question_types[i % len(question_types)]
        question["type"] = q_type
        
        # Add difficulty if specified
        if difficulty:
            question["difficulty"] = difficulty
        
        # Generate options for multiple choice questions
        if q_type == "multiple_choice" and "options" not in question:
            options = generate_options(question["question"], question.get("answer", ""), 
                                      question.get("context", ""))
            question["options"] = options
    
    # Return the best questions
    return filtered_questions[:num_questions] if filtered_questions else []

def generate_template_questions(chunks, num_questions):
    """Generate simple template-based questions when all else fails"""
    questions = []
    
    # Extract some keywords from the text
    all_text = " ".join(chunks)
    
    # Find potential subjects (capitalized words)
    subjects = re.findall(r'\b[A-Z][a-zA-Z]{3,}\b', all_text)
    
    # If we don't have subjects, extract frequent words
    if not subjects:
        words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text.lower())
        from collections import Counter
        word_counts = Counter(words)
        subjects = [word for word, count in word_counts.most_common(5) if count > 1]
    
    # Ensure we have some subjects
    if not subjects:
        subjects = ["this topic", "the document", "the text"]
    
    # Template questions
    templates = [
        "What is the main purpose of {}?",
        "How would you describe {}?",
        "What are the key characteristics of {}?",
        "What is the significance of {} in this context?",
        "How does {} relate to the main topic?",
        "What role does {} play in the document?",
        "How would you explain {} to someone unfamiliar with the topic?",
        "What is the relationship between {} and the overall subject?",
    ]
    
    # Generate questions from templates
    for i in range(min(num_questions, len(subjects) * len(templates))):
        subject = subjects[i % len(subjects)]
        template = templates[(i // len(subjects)) % len(templates)]
        
        question_text = template.format(subject)
        
        questions.append({
            "question": question_text,
            "answer": f"Based on the document, {subject.lower()} appears to be a key concept worth understanding.",
            "context": chunks[0] if chunks else "",
            "source": "template"
        })
    
    return questions

def is_good_question(question_text):
    """Check if a question meets basic quality criteria"""
    if not question_text or len(question_text.strip()) < 10:
        return False
    
    # Check if it ends with a question mark
    if not question_text.strip().endswith('?'):
        return False
    
    # Check if it contains interrogative words
    interrogatives = ['what', 'why', 'how', 'when', 'where', 'which', 'who', 'whose', 'whom']
    has_interrogative = any(q in question_text.lower().split() for q in interrogatives)
    
    return has_interrogative

def generate_options(question, correct_answer, context, num_options=4):
    """Generate options for multiple choice questions"""
    if not correct_answer:
        return ["Option A", "Option B", "Option C", "Option D"]
    
    # Create distractor options
    options = [correct_answer]
    
    # Extract key terms from context
    words = re.findall(r'\b[A-Za-z][a-z]{3,}\b', context)
    words = [w for w in words if w.lower() not in correct_answer.lower()]
    
    # If not enough words, create generic options
    if len(words) < num_options - 1:
        while len(options) < num_options:
            options.append(f"Option {len(options)}")
        return options
    
    # Select random words as distractors
    random.shuffle(words)
    for word in words:
        if len(options) < num_options:
            options.append(word)
    
    # Shuffle options so correct answer isn't always first
    random.shuffle(options)
    
    return options