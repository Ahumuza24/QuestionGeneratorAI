import torch
import re
import random
import os
import json
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, pipeline, Trainer, TrainingArguments
import numpy as np
from torch.utils.data import Dataset, DataLoader
from app.utils.document_processor import extract_text_from_pdf

# Initialize tokenizer and model
model_name = 'valhalla/t5-base-e2e-qg'
tokenizer = T5Tokenizer.from_pretrained(
    model_name,
    model_max_length=512
)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Define paths for saving fine-tuned models
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

FINE_TUNED_MODEL_PATH = os.path.join(MODELS_DIR, 'fine_tuned_question_generator')

# Check if fine-tuned model exists and load it if available
if os.path.exists(FINE_TUNED_MODEL_PATH):
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(FINE_TUNED_MODEL_PATH)
        print(f"Loaded fine-tuned model from {FINE_TUNED_MODEL_PATH}")
    except Exception as e:
        print(f"Error loading fine-tuned model: {e}. Using base model instead.")

# Check for CUDA availability
device = 0 if torch.cuda.is_available() else -1

# Initialize the pipelines
question_generator = pipeline(
    'text2text-generation',
    model=model,
    tokenizer=tokenizer,
    device=device
)
answer_generator = pipeline('question-answering', model='deepset/roberta-base-squad2', device=device)

# Dataset class for fine-tuning
class QuestionGenerationDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        example = self.examples[idx]
        context = example["context"]
        question = example["question"]
        
        # Format the input for question generation
        input_text = f"generate question: {context}"
        
        # Tokenize input and target
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        target_encoding = self.tokenizer(
            question,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = input_encoding["input_ids"].squeeze()
        attention_mask = input_encoding["attention_mask"].squeeze()
        labels = target_encoding["input_ids"].squeeze()
        
        # Replace padding tokens in labels with -100 so they're ignored in loss calculation
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def train_model_on_examples(examples, output_dir=FINE_TUNED_MODEL_PATH, epochs=3, batch_size=8):
    """
    Fine-tune the question generation model on example questions
    
    Args:
        examples (list): List of dicts with 'context' and 'question' keys
        output_dir (str): Directory to save the fine-tuned model
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        
    Returns:
        str: Path to the saved model
    """
    global model, tokenizer, question_generator
    
    print(f"Starting fine-tuning on {len(examples)} examples")
    
    # Prepare dataset
    dataset = QuestionGenerationDataset(examples, tokenizer)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=len(dataset) // batch_size,
        save_total_limit=2,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=len(dataset) // (batch_size * 2),
        warmup_steps=500,
        weight_decay=0.01,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    # Train the model
    trainer.train()
    
    # Save the fine-tuned model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Update the global model and pipeline with fine-tuned version
    model = trainer.model
    question_generator = pipeline(
        'text2text-generation',
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    
    print(f"Model fine-tuning complete. Saved to {output_dir}")
    return output_dir

def prepare_examples_from_question_paper(paper_path):
    """
    Extract training examples from a question paper file
    
    Args:
        paper_path (str): Path to the question paper file (JSON format)
        
    Returns:
        list: Examples in the format needed for training
    """
    examples = []
    
    try:
        with open(paper_path, 'r', encoding='utf-8') as f:
            paper_data = json.load(f)
            
        # Expected format: list of question objects with 'context', 'question', 'answer' fields
        for item in paper_data:
            if 'context' in item and 'question' in item:
                examples.append({
                    'context': item['context'],
                    'question': item['question']
                })
    except Exception as e:
        print(f"Error processing question paper file: {e}")
    
    return examples

def load_question_papers_from_directory(directory_path):
    """
    Load all question paper examples from a directory
    
    Args:
        directory_path (str): Path to directory containing question paper files
        
    Returns:
        list: Combined examples from all papers
    """
    all_examples = []
    
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return all_examples
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            examples = prepare_examples_from_question_paper(file_path)
            all_examples.extend(examples)
            print(f"Loaded {len(examples)} examples from {filename}")
    
    return all_examples

def generate_questions(text, num_questions=5, question_type='both', difficulty='medium'):
    """
    Generate questions from document text
    
    Args:
        text (str): Document text
        num_questions (int): Number of questions to generate
        question_type (str): Type of questions to generate ('multiple_choice', 'structured', or 'both')
        difficulty (str): Difficulty level ('easy', 'medium', 'hard')
        
    Returns:
        list: List of generated questions with answers
    """
    # Validate and cap number of questions
    num_questions = min(max(1, int(num_questions)), 50)
    
    # Preprocess the text
    sentences = sent_tokenize(text)
    sentences = [s.strip() for s in sentences if len(s.split()) > 5]
    
    if len(sentences) < 3:
        return []
    
    # Use TF-IDF to identify important sentences
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    importance_scores = tfidf_matrix.sum(axis=1).A1
    
    # Create a list of (sentence, score) tuples and sort by score
    sentence_scores = list(zip(sentences, importance_scores))
    sentence_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Select more sentences than needed to ensure we can reach the requested question count
    # We'll select twice as many sentences as questions to have sufficient material
    extra_factor = 3
    selected_sentences = [s[0] for s in sentence_scores[:num_questions * extra_factor]]
    random.shuffle(selected_sentences)
    
    # Generate questions based on the type
    if question_type == 'multiple_choice':
        return ensure_question_count(
            generate_multiple_choice_questions(selected_sentences, num_questions, difficulty),
            num_questions,
            question_type,
            selected_sentences,
            difficulty
        )
    elif question_type == 'structured':
        return ensure_question_count(
            generate_structured_questions(selected_sentences, num_questions, difficulty),
            num_questions,
            question_type,
            selected_sentences,
            difficulty
        )
    else:  # 'both'
        mc_count = num_questions // 2
        structured_count = num_questions - mc_count
        
        mc_questions = generate_multiple_choice_questions(
            selected_sentences[:len(selected_sentences) // 2], 
            mc_count, 
            difficulty
        )
        
        structured_questions = generate_structured_questions(
            selected_sentences[len(selected_sentences) // 2:], 
            structured_count, 
            difficulty
        )
        
        # Ensure we have the exact count requested for each type
        mc_questions = ensure_question_count(
            mc_questions,
            mc_count,
            'multiple_choice',
            selected_sentences,
            difficulty
        )
        
        structured_questions = ensure_question_count(
            structured_questions,
            structured_count,
            'structured',
            selected_sentences,
            difficulty
        )
        
        return mc_questions + structured_questions

def ensure_question_count(questions, target_count, question_type, sentences, difficulty):
    """
    Ensure we have exactly the requested number of questions.
    If we have too few, generate more. If we have too many, trim.
    
    Args:
        questions (list): Currently generated questions
        target_count (int): Number of questions we need
        question_type (str): Type of questions to generate
        sentences (list): Available sentences to generate from
        difficulty (str): Difficulty level
        
    Returns:
        list: List with exactly target_count questions
    """
    if len(questions) == target_count:
        return questions
        
    # If we have too many questions, trim to the target count
    if len(questions) > target_count:
        return questions[:target_count]
        
    # If we have too few questions, we need to generate more
    additional_needed = target_count - len(questions)
    
    # First, try to use any remaining sentences we haven't tried yet
    used_contexts = [q['context'] for q in questions]
    remaining_sentences = [s for s in sentences if s not in used_contexts]
    
    # If we have no remaining sentences, we'll reuse some, preferring ones not already used
    if not remaining_sentences:
        # Reuse sentences, prioritizing those we haven't used yet
        remaining_sentences = sentences
        
    # Retry generation with remaining sentences
    additional_questions = []
    if question_type == 'multiple_choice':
        additional_questions = generate_multiple_choice_questions(
            remaining_sentences, 
            additional_needed, 
            difficulty
        )
    else:  # structured
        additional_questions = generate_structured_questions(
            remaining_sentences, 
            additional_needed, 
            difficulty
        )
    
    # If we still don't have enough, fall back to generating generic questions
    still_needed = target_count - (len(questions) + len(additional_questions))
    if still_needed > 0:
        for i in range(still_needed):
            context = random.choice(sentences)
            if question_type == 'multiple_choice':
                # Generic multiple choice question as fallback
                answer = "Option A"
                additional_questions.append({
                    'question': f"Question {len(questions) + len(additional_questions) + 1} about: {context[:50]}...?",
                    'answer': answer,
                    'options': [answer, "Option B", "Option C", "Option D"],
                    'context': context,
                    'type': 'multiple_choice',
                    'confidence': 1.0
                })
            else:
                # Generic structured question as fallback
                additional_questions.append({
                    'question': f"Question {len(questions) + len(additional_questions) + 1} about: {context[:50]}...?",
                    'answer': "Answer based on the context.",
                    'context': context,
                    'type': 'structured',
                    'confidence': 1.0
                })
    
    # Combine original questions with additional ones
    return questions + additional_questions[:additional_needed]

def generate_structured_questions(sentences, num_questions, difficulty):
    """Generate structured questions from sentences"""
    questions = []
    num_to_generate = min(num_questions, len(sentences))
    
    # Increase the number of attempts per sentence to ensure we get more questions
    max_attempts_per_sentence = 4
    max_sentences_to_try = min(len(sentences), num_questions * 2)
    
    # Try generating from each sentence until we have enough questions
    for i in range(max_sentences_to_try):
        if i >= len(sentences):
            break
            
        if len(questions) >= num_questions:
            break
            
        context = sentences[i]
        question_text = None
        answer = None
        
        # Modified prompt to be more explicit about single question generation
        prompt = f"Generate exactly one question from this text. The question must be answerable from the text. Do not generate multiple questions: {context}"
        
        # Try multiple times per sentence to get a valid question
        for attempt in range(max_attempts_per_sentence):
            generated_output = question_generator(
                prompt, 
                max_length=64, 
                num_return_sequences=1,
                do_sample=True,
                temperature=0.6, # Slightly lower temperature for more focused output
                top_p=0.85,
                no_repeat_ngram_size=3,
                early_stopping=True # Enable early stopping
            )[0]['generated_text']
        
            # Clean and validate the question
            cleaned_question = clean_question_text(generated_output)
            if cleaned_question and '?' in cleaned_question:
                # Verify the answer exists in the context
                potential_answer = answer_generator(
                    question=cleaned_question,
                    context=context
                )
                
                # Accept questions with lower confidence if we're struggling to generate enough
                confidence_threshold = 0.5 if len(questions) < num_questions / 2 else 0.7
                
                if potential_answer['score'] > confidence_threshold:
                    question_text = cleaned_question
                    answer = potential_answer
                    break # Got a good question, exit retry loop
        
        # Only add new questions if they're different from existing ones
        if question_text and answer:
            # Check if this question is too similar to ones we already have
            is_duplicate = False
            for existing_q in questions:
                # Basic similarity check - if questions share many words
                existing_words = set(existing_q['question'].lower().split())
                new_words = set(question_text.lower().split())
                overlap = len(existing_words.intersection(new_words)) / len(existing_words.union(new_words))
                
                if overlap > 0.7:  # More than 70% word overlap
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                questions.append({
                    'question': question_text,
                    'answer': answer['answer'],
                    'context': context,
                    'type': 'structured',
                    'confidence': answer['score']
                })
    
    # Return the number of questions requested, or as many as we could generate
    return questions[:num_questions]

def generate_multiple_choice_questions(sentences, num_questions, difficulty):
    """Generate multiple-choice questions from sentences"""
    questions = []
    
    # Increase the number of attempts per sentence to ensure we get more questions
    max_attempts_per_sentence = 4
    max_sentences_to_try = min(len(sentences), num_questions * 2)
    
    # Try generating from each sentence until we have enough questions
    for i in range(max_sentences_to_try):
        if i >= len(sentences):
            break
            
        if len(questions) >= num_questions:
            break
            
        context = sentences[i]
        
        # Modified prompt to ensure relevance and clarity about generating MC questions
        prompt = f"Based on this specific text, generate one multiple choice question with 4 options that can be answered directly from the text: {context}"
        
        # Try multiple times per sentence to get a valid question
        for attempt in range(max_attempts_per_sentence):
            question_text = question_generator(
                prompt, 
                max_length=64, 
                num_return_sequences=1,
                do_sample=True,
                temperature=0.6,  # Reduced temperature
                top_p=0.85,
                no_repeat_ngram_size=3
            )[0]['generated_text']
        
            # Clean and validate the question
            cleaned_question = clean_question_text(question_text)
            if cleaned_question and '?' in cleaned_question:
                # Verify the answer exists in the context
                answer_result = answer_generator(
                    question=cleaned_question, 
                    context=context
                )
                
                # Accept questions with lower confidence if we're struggling to generate enough
                confidence_threshold = 0.5 if len(questions) < num_questions / 2 else 0.7
                
                if answer_result['score'] > confidence_threshold:
                    question_text = cleaned_question
                    correct_answer = answer_result['answer']
                    
                    # Check if this question is too similar to ones we already have
                    is_duplicate = False
                    for existing_q in questions:
                        # Basic similarity check - if questions share many words
                        existing_words = set(existing_q['question'].lower().split())
                        new_words = set(question_text.lower().split())
                        overlap = len(existing_words.intersection(new_words)) / len(existing_words.union(new_words))
                        
                        if overlap > 0.7:  # More than 70% word overlap
                            is_duplicate = True
                            break
                            
                    if not is_duplicate:
                        # Generate distractors based on the context
                        distractors = generate_distractors(context, correct_answer, difficulty)
                        
                        # Combine correct answer and distractors
                        options = [correct_answer] + distractors[:3]
                        random.shuffle(options)
                        
                        questions.append({
                            'question': question_text,
                            'answer': correct_answer,
                            'options': options,
                            'context': context,
                            'type': 'multiple_choice',
                            'confidence': answer_result['score']
                        })
                        
                        # Break out of the retry loop since we got a good question
                        break
    
    # Return the number of questions requested, or as many as we could generate
    return questions[:num_questions]

def generate_distractors(context, correct_answer, difficulty):
    """Generate wrong options for multiple choice questions"""
    # Adjust temperature based on difficulty
    temp = 0.6  # Reduced default temperature
    if difficulty == 'easy':
        temp = 0.5
    elif difficulty == 'hard':
        temp = 0.7
    
    # Modified prompt to ensure relevant distractors
    prompt = f"""Based on this text: {context}
Generate 3 plausible but incorrect answer options that are related to the context.
Correct answer: {correct_answer}
The options should be different from: {correct_answer}"""
    
    results = question_generator(
        prompt,
        max_length=128,
        num_return_sequences=1,
        do_sample=True,
        temperature=temp,
        top_p=0.85
    )[0]['generated_text']
    
    # Parse and clean distractors
    distractors = []
    for line in results.split('\n'):
        line = line.strip()
        if line and line != correct_answer and not line.startswith(("Context:", "Correct answer:", "Generate")):
            # Remove any numbering or bullet points
            clean_line = re.sub(r'^[\d\-\.\)\•\*]+\s*', '', line)
            if clean_line and clean_line not in distractors and clean_line != correct_answer:
                distractors.append(clean_line)
    
    # If we didn't get enough distractors, generate some based on the context
    while len(distractors) < 3:
        # Use answer_generator to find other entities in the context
        probe_question = f"What is another {correct_answer.split()[0]} mentioned in the text?"
        probe_answer = answer_generator(question=probe_question, context=context)
        if probe_answer['answer'] and probe_answer['answer'] != correct_answer:
            distractors.append(probe_answer['answer'])
        else:
            generic = f"Alternative option {len(distractors) + 1}"
            distractors.append(generic)
    
    return distractors[:3]

def clean_question_text(text):
    """Clean up generated question text to ensure only one question is returned"""
    # First, split on <sep> token if present and take the first part
    if '<sep>' in text:
        text = text.split('<sep>')[0].strip()
    
    # Remove common prefixes and formatting that might indicate multiple items
    text = re.sub(r'^(Q:|Question:|A:|Answer:|\\d+[\\.\\)]|\\-|\\*)\\s*', '', text, flags=re.IGNORECASE).strip()
    
    # Split by newline, periods followed by space (if not ending the string), or question marks (if not ending)
    # Prioritize splitting by newline as it often separates distinct generated questions
    potential_questions = re.split(r'\\n+|(?<!\\w)\\.\\s+(?=\\w)|\\?(?!$)', text)
    
    # Find the first valid-looking question
    question = None
    for pq in potential_questions:
        pq = pq.strip()
        if len(pq.split()) >= 3 and '?' in pq: # Check length and presence of question mark
            question = pq
            break
        elif len(pq.split()) >= 3 and not '?' in pq: # Check if it's just missing a question mark
            question = pq + '?'
            break
    
    if not question:
        # Fallback if splitting didn't find a clear question
        # Take the original text if it looks plausible, otherwise return None
        if len(text.split()) >= 3:
            question = text
        else:
            return None
    
    # Further cleanup on the selected question
    # Remove any remaining leading numbering/bullets
    question = re.sub(r'^[\\d\\-\\.\\)\\•\\*]+\\s*', '', question).strip()
    
    # Ensure it ends with a single question mark
    question = question.rstrip('. ')
    if not question.endswith('?'):
        question += '?'
    
    # Handle cases where multiple question marks might still exist
    if question.count('?') > 1:
        question = question.split('?')[0] + '?'
    
    # Final length check
    if len(question.split()) < 3:
        return None
    
    return question

def parse_questions_from_pdf(pdf_path):
    """
    Extract questions and answers from a PDF question paper
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        list: Extracted examples in the format needed for training
    """
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    
    # Split text into sections/paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    
    # Patterns to identify questions and answers
    question_patterns = [
        # Common question patterns
        r'(?i)^\s*(\d+\.|\d+\)|\d+\s+|[A-Z]\.|\([a-z]\))\s*(.+\?)',  # Numbered questions with question mark
        r'(?i)^\s*(?:Question|Q)[\s:\.]+(.*\?)',  # Questions labeled with "Question:" or "Q:"
        r'(?i)(.+\?)\s*(?:\n|$)'  # Any line ending with a question mark
    ]
    
    # Patterns to identify answers 
    answer_patterns = [
        # Common answer patterns
        r'(?i)^\s*(?:Answer|A)[\s:\.]+(.*)',  # Answers labeled with "Answer:" or "A:"
        r'(?i)^\s*(?:Solution|Sol)[\s:\.]+(.*)',  # Solutions
        r'(?i)^\s*(?:Correct answer)[\s:\.]+(.*)'  # Correct answer 
    ]
    
    examples = []
    current_context = ""
    current_question = None
    
    # Process each paragraph
    for para in paragraphs:
        para = para.strip()
        if len(para) < 10:  # Skip very short paragraphs
            continue
            
        # If this paragraph doesn't look like a question or answer, treat it as context
        is_question = False
        is_answer = False
        
        # Check if this paragraph contains a question
        for pattern in question_patterns:
            match = re.search(pattern, para)
            if match:
                # If we already have a question without an answer, save the previous question
                if current_question and current_context:
                    examples.append({
                        'context': current_context,
                        'question': current_question,
                        'answer': "Unable to extract answer automatically"
                    })
                
                # Get the question text
                question_text = match.group(2) if len(match.groups()) > 1 else match.group(1)
                current_question = question_text.strip()
                current_context = para  # Use the paragraph as context
                is_question = True
                break
                
        # Check if this paragraph contains an answer to the current question
        if current_question and not is_question:
            for pattern in answer_patterns:
                match = re.search(pattern, para)
                if match:
                    answer_text = match.group(1)
                    
                    # Add the example with question, answer and context
                    examples.append({
                        'context': current_context,
                        'question': current_question,
                        'answer': answer_text.strip()
                    })
                    
                    # Reset for next question
                    current_question = None
                    current_context = ""
                    is_answer = True
                    break
                    
        # If it's neither a question nor an answer, and we have a current question,
        # append this paragraph to the context in case it contains information for the answer
        if current_question and not is_question and not is_answer:
            current_context += " " + para
    
    # If we have a pending question without an answer at the end
    if current_question and current_context:
        examples.append({
            'context': current_context,
            'question': current_question,
            'answer': "Unable to extract answer automatically"
        })
    
    return examples

def extract_multiple_choice_from_pdf(pdf_path):
    """
    Extract multiple-choice questions from a PDF file
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        list: Extracted multiple-choice examples
    """
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    
    # Split text into potential question blocks
    # Looking for patterns like "1. Question text" followed by options
    question_blocks = re.split(r'\n\s*(?:\d+\.|\(\d+\))', text)
    
    examples = []
    
    for block in question_blocks:
        block = block.strip()
        if len(block) < 20:  # Skip short blocks
            continue
            
        # Extract the question (assuming it ends with a question mark)
        question_match = re.search(r'(.*?\?)', block)
        if not question_match:
            continue
            
        question = question_match.group(1).strip()
        
        # Extract options - look for patterns like "A. Option text" or "a) Option text"
        options = []
        options_text = block[question_match.end():].strip()
        
        option_matches = re.finditer(r'(?:^|\n)\s*([A-Za-z])[\.:\)][ \t]*(.*?)(?=\n\s*[A-Za-z][\.:\)]|$)', options_text)
        for match in option_matches:
            option_label = match.group(1)
            option_text = match.group(2).strip()
            options.append((option_label, option_text))
            
        # If we found a question and at least 2 options
        if question and len(options) >= 2:
            # Look for the correct answer indicator (often marked with *)
            correct_answer = None
            for label, text in options:
                if '*' in text or '(correct)' in text.lower():
                    correct_answer = text.replace('*', '').replace('(correct)', '').strip()
                    break
                    
            # If no marked correct answer, try to find it in the text
            if not correct_answer and len(options) > 0:
                # Look for patterns like "Answer: B" or "Correct option: C"
                answer_match = re.search(r'(?:Answer|Correct)[:\s]+([A-Za-z])', block)
                if answer_match:
                    correct_label = answer_match.group(1)
                    for label, text in options:
                        if label.upper() == correct_label.upper():
                            correct_answer = text
                            break
            
            # If still no correct answer, use the first option as a fallback
            if not correct_answer and len(options) > 0:
                correct_answer = options[0][1]
                
            # Format options for the training example
            option_texts = [text for _, text in options]
            
            examples.append({
                'context': block,
                'question': question,
                'answer': correct_answer,
                'options': option_texts,
                'type': 'multiple_choice'
            })
    
    return examples