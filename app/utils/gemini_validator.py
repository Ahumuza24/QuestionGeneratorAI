"""
Gemini-based validator for ensuring question quality
"""
import os
import json
import time
import random
from typing import List, Dict, Any

# Check if the Gemini package is available
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    print("Google GenerativeAI package not found. Install with 'pip install google-generativeai'")

# Track Gemini availability state
GEMINI_AVAILABLE = False
GEMINI_LAST_CHECK_TIME = 0
GEMINI_CHECK_INTERVAL = 300  # 5 minutes between checks
MAX_RETRIES = 3
BACKOFF_FACTOR = 2  # Exponential backoff factor

# Configure the API key if Gemini is available
if HAS_GEMINI:
    API_KEY = os.environ.get('GEMINI_API_KEY')
    if not API_KEY:
        print("Warning: GEMINI_API_KEY environment variable not set. Gemini validation will not work.")
    else:
        try:
            # Configure the Gemini API
            genai.configure(api_key=API_KEY)

            # List available models to debug
            available_models = genai.list_models()
           
            # Set the correct model name based on available models
            MODEL_NAMES = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
            MODEL_NAME = None
            
            for model_name in MODEL_NAMES:
                if any(model_name in model.name for model in available_models):
                    MODEL_NAME = model_name
                    print(f"Selected Gemini model: {MODEL_NAME}")
                    GEMINI_AVAILABLE = True
                    break
                    
            if MODEL_NAME is None:
                print("No compatible Gemini model found. Check API key and available models.")
                HAS_GEMINI = False
                GEMINI_AVAILABLE = False
                
        except Exception as e:
            print(f"Error configuring Gemini API: {e}")
            HAS_GEMINI = False
            GEMINI_AVAILABLE = False

# Configure the model
TEMPERATURE = 0.2  # Low temperature for more consistent results
TOP_P = 0.95
TOP_K = 40

def get_model():
    """Get the Gemini model if API key is available"""
    if not HAS_GEMINI or not API_KEY:
        return None
    
    try:
        return genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config={
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "top_k": TOP_K
            }
        )
    except Exception as e:
        print(f"Error initializing Gemini model: {e}")
        return None

def test_gemini_connection(timeout=3):
    """
    Test if the Gemini API is available and responding.
    Uses a simple request with timeout to check connectivity.
    
    Args:
        timeout: Maximum time to wait for response in seconds
    
    Returns:
        bool: True if Gemini is available, False otherwise
    """
    print("Testing Gemini connection")
    
    # Check if the required modules and API key are available
    if not HAS_GEMINI or not API_KEY:
        print("Gemini not available: missing modules or API key")
        return False
    
    try:
        import google.generativeai as genai
        import threading
        import time
        
        result = None
        error = None
        
        def call_api():
            nonlocal result, error
            try:
                genai.configure(api_key=API_KEY)
                model = genai.GenerativeModel(MODEL_NAME)
                result = model.generate_content("Return the word 'available' if you can read this.")
            except Exception as e:
                error = e
                print(f"Gemini API error during test: {e}")
        
        # Use threading to implement timeout
        thread = threading.Thread(target=call_api)
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            print("Gemini API test timed out")
            return False
        
        if error:
            return False
            
        # Check if we got a reasonable response
        if result and result.text.strip().lower() == "available":
            print("Gemini API test successful")
            return True
        else:
            print(f"Gemini API test returned unexpected response: {result.text if result else 'None'}")
            return False
            
    except Exception as e:
        print(f"Error testing Gemini connection: {e}")
        return False

def validate_question(question: Dict[str, Any], fast_mode: bool = True, retry_on_failure: bool = True) -> Dict[str, Any]:
    """
    Validate a single question using Gemini with fast mode option and retry logic
    
    Args:
        question: Dictionary containing question data
        fast_mode: Whether to use faster validation (less detailed)
        retry_on_failure: Whether to retry on failure before falling back
        
    Returns:
        Dictionary with validation results added
    """
    # Check if Gemini is available before attempting validation
    if not test_gemini_connection():
        # Add fallback validation result directly
        question['gemini_validation'] = {
            'is_valid': True,  # Default to valid if no model
            'validation_score': 0.7,
            'educational_value': 0.7,
            'feedback': "Gemini validation not available, using T5 fallback instead",
            'used_fallback': True
        }
        return question
    
    model = get_model()
    if not model:
        question['gemini_validation'] = {
            'is_valid': True,  # Default to valid if no model
            'validation_score': 0.7,
            'educational_value': 0.7,
            'feedback': "Gemini model not available, using T5 fallback instead",
            'used_fallback': True
        }
        return question
    
    # Extract relevant information from the question
    question_text = question.get('question', '')
    context = question.get('context', '')
    answer = question.get('answer', '')
    question_type = question.get('type', 'structured')
    
    # OPTIMIZATION: For fast mode, use a shorter context
    if fast_mode and len(context) > 1000:
        # Get context around the answer if possible
        if answer and answer in context:
            answer_pos = context.find(answer)
            start_pos = max(0, answer_pos - 300)
            end_pos = min(len(context), answer_pos + len(answer) + 300)
            context = context[start_pos:end_pos]
        else:
            # Just take the first part of the context
            context = context[:1000]
    
    # For multiple choice, include options
    options_text = ""
    if question_type == 'multiple_choice' and 'options' in question:
        for i, option in enumerate(question['options']):
            options_text += f"Option {chr(65+i)}: {option}\n"
    
    # OPTIMIZATION: Simplified prompt for fast mode
    if fast_mode:
        validation_prompt = f"""
        Rate this educational question on a scale of 0.0 to 1.0 for:
        - Validity: Is it answerable from the context?
        - Educational Value: Does it test important concepts?
        
        Context: "{context}"
        Question: "{question_text}"
        {options_text}
        Answer: "{answer}"
        
        Return ONLY a JSON object with these exact keys:
        {{"overall_score": float, "is_valid": boolean, "feedback": "brief feedback"}}
        """
    else:
        # Use the original comprehensive prompt
        validation_prompt = f"""
        You are an educational question validator. Analyze this question for quality and accuracy.
        
        Context from educational material:
        ```
        {context}
        ```
        
        Question: {question_text}
        
        {options_text}
        
        Answer/Expected Answer: {answer}
        
        Please evaluate this question based on the following criteria and provide a structured JSON response:
        
        1. Validity: Is the question answerable from the given context? (0.0-1.0)
        2. Relevance: Does the question test important concepts from the context? (0.0-1.0)
        3. Clarity: Is the question clearly worded and unambiguous? (0.0-1.0)
        4. Educational Value: Does the question promote critical thinking and comprehension? (0.0-1.0)
        5. Difficulty: Is the question at an appropriate difficulty level? (0.0-1.0)
        6. Accuracy: Is the expected answer correct based on the context? (0.0-1.0)
        
        For multiple choice questions, also evaluate:
        7. Quality of Distractors: Are the wrong options plausible but clearly incorrect? (0.0-1.0)
        
        Return your evaluation as a JSON object with these exact keys (nothing else):
        {{"validity": float, "relevance": float, "clarity": float, "educational_value": float, "difficulty": float, "accuracy": float, "quality_of_distractors": float, "overall_score": float, "is_valid": boolean, "feedback": "brief feedback", "improved_question": "improved version of question"}}
        
        If any field is not applicable (e.g., quality_of_distractors for structured questions), set it to null.
        """
    
    # Implement retry logic with exponential backoff
    retries = 0
    while retries <= (MAX_RETRIES if retry_on_failure else 0):
        try:
            # Send to Gemini for validation with timeout
            import threading
            
            result = None
            error = None
            
            def call_api():
                nonlocal result, error
                try:
                    result = model.generate_content(validation_prompt)
                except Exception as e:
                    error = e
            
            # Start API call in a thread
            thread = threading.Thread(target=call_api)
            thread.start()
            
            # Wait for 5 seconds max (adjust timeout based on retry count)
            timeout = 5.0 + (retries * 2.0)  # Increase timeout with each retry
            thread.join(timeout)
            
            if thread.is_alive():
                print(f"Gemini validation timed out (retry {retries}/{MAX_RETRIES})")
                if retries == MAX_RETRIES:
                    # Return default values on final timeout
                    question['gemini_validation'] = {
                        'is_valid': True,
                        'validation_score': 0.7,
                        'educational_value': 0.7,
                        'feedback': "Validation timed out, using T5 fallback",
                        'used_fallback': True
                    }
                    return question
                else:
                    # Prepare for retry
                    retries += 1
                    backoff_time = BACKOFF_FACTOR ** retries + random.uniform(0, 1)
                    print(f"Retrying in {backoff_time:.2f} seconds...")
                    time.sleep(backoff_time)
                    continue
            
            if error:
                if retries == MAX_RETRIES:
                    raise error
                else:
                    # Prepare for retry
                    print(f"Error during validation (retry {retries}/{MAX_RETRIES}): {error}")
                    retries += 1
                    backoff_time = BACKOFF_FACTOR ** retries + random.uniform(0, 1)
                    print(f"Retrying in {backoff_time:.2f} seconds...")
                    time.sleep(backoff_time)
                    continue
            
            if not result:
                if retries == MAX_RETRIES:
                    raise Exception("No result from Gemini")
                else:
                    # Prepare for retry
                    print(f"No result from Gemini (retry {retries}/{MAX_RETRIES})")
                    retries += 1
                    backoff_time = BACKOFF_FACTOR ** retries + random.uniform(0, 1)
                    print(f"Retrying in {backoff_time:.2f} seconds...")
                    time.sleep(backoff_time)
                    continue
            
            # Extract JSON from response
            result_text = result.text
            
            # Parse JSON (simplified for speed)
            try:
                # Quick attempt to parse
                validation_result = json.loads(result_text)
            except json.JSONDecodeError:
                # Fallback regex extraction
                import re
                json_match = re.search(r'{[\s\S]*}', result_text)
                if json_match:
                    try:
                        validation_result = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        validation_result = {}
                else:
                    validation_result = {}
            
            # Ensure we have minimum required fields
            if "overall_score" not in validation_result:
                if fast_mode:
                    # In fast mode, just estimate a score
                    validation_result["overall_score"] = 0.7
                else:
                    # Calculate from detailed scores
                    scores = [
                        validation_result.get("validity", 0.0),
                        validation_result.get("relevance", 0.0),
                        validation_result.get("clarity", 0.0), 
                        validation_result.get("educational_value", 0.0),
                        validation_result.get("accuracy", 0.0)
                    ]
                    
                    valid_scores = [s for s in scores if isinstance(s, (int, float))]
                    if valid_scores:
                        validation_result["overall_score"] = sum(valid_scores) / len(valid_scores)
                    else:
                        validation_result["overall_score"] = 0.5
            
            # Determine if question is valid
            validation_result["is_valid"] = validation_result.get("overall_score", 0.0) >= 0.6
            
            # Flag that we used Gemini (not the fallback)
            validation_result["used_fallback"] = False
            
            # Add validation results to question
            question['gemini_validation'] = validation_result
            
            # If we get here, we succeeded
            return question
            
        except Exception as e:
            if retries == MAX_RETRIES or not retry_on_failure:
                print(f"Error during Gemini validation after {retries} retries: {e}")
                # Return original question with minimal validation
                question['gemini_validation'] = {
                    'is_valid': True,  # Default to valid on error
                    'validation_score': 0.7,
                    'educational_value': 0.7,
                    'feedback': f"Error during validation: {str(e)}, using T5 fallback instead",
                    'used_fallback': True
                }
                
                # Mark Gemini as unavailable to avoid further failed calls
                global GEMINI_AVAILABLE
                GEMINI_AVAILABLE = False
                
                return question
            else:
                # Prepare for retry
                print(f"Error during validation (retry {retries}/{MAX_RETRIES}): {e}")
                retries += 1
                backoff_time = BACKOFF_FACTOR ** retries + random.uniform(0, 1)
                print(f"Retrying in {backoff_time:.2f} seconds...")
                time.sleep(backoff_time)

# OPTIMIZATION: Add a batch validation function with strict time limits
def fast_batch_validate(questions: List[Dict[str, Any]], max_time_seconds: int = 60) -> List[Dict[str, Any]]:
    """
    Validate questions with a strict time limit
    
    Args:
        questions: List of questions to validate
        max_time_seconds: Maximum time to spend on validation
    
    Returns:
        List of validated questions
    """
    import time
    start_time = time.time()
    validated = []
    
    for question in questions:
        # Check if we're out of time
        if time.time() - start_time > max_time_seconds:
            # Add remaining questions without validation
            for q in questions[len(validated):]:
                q['gemini_validation'] = {
                    'is_valid': True,
                    'validation_score': 0.7,
                    'educational_value': 0.7,
                    'feedback': "Skipped validation due to time constraints"
                }
                validated.append(q)
            break
        
        # Validate this question (fast mode)
        validated.append(validate_question(question, fast_mode=True))
    
    return validated

def batch_validate_questions(questions: List[Dict[str, Any]], batch_size: int = 5) -> List[Dict[str, Any]]:
    """
    Validate a batch of questions using Gemini
    
    Args:
        questions: List of question dictionaries
        batch_size: Number of questions to validate at once
        
    Returns:
        List of questions with validation results added
    """
    validated_questions = []
    
    # Process in small batches to avoid rate limits
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i+batch_size]
        
        # Validate each question in the batch
        for question in batch:
            validated_question = validate_question(question)
            validated_questions.append(validated_question)
    
    return validated_questions

def improve_question_with_gemini(question: Dict[str, Any]) -> Dict[str, Any]:
    """
    Improve a question using Gemini's suggestions
    
    Args:
        question: Dictionary containing question data
        
    Returns:
        Improved question dictionary
    """
    model = get_model()
    if not model:
        return question
    
    # Extract relevant information from the question
    question_text = question.get('question', '')
    context = question.get('context', '')
    answer = question.get('answer', '')
    question_type = question.get('type', 'structured')
    
    # For multiple choice, include options
    options_text = ""
    if question_type == 'multiple_choice' and 'options' in question:
        for i, option in enumerate(question['options']):
            options_text += f"Option {chr(65+i)}: {option}\n"
    
    # Prepare prompt for Gemini
    improvement_prompt = f"""
    You are an educational question improvement expert. Rewrite and enhance this question to make it more effective.
    
    Context from educational material:
    ```
    {context}
    ```
    
    Original Question: {question_text}
    
    {options_text}
    
    Expected Answer: {answer}
    
    Please improve this question by:
    1. Making it more clear and precise
    2. Ensuring it tests important concepts from the material
    3. Increasing its educational value and critical thinking requirements
    4. Ensuring it remains answerable from the given context
    5. For multiple choice, making distractors more plausible yet clearly incorrect
    
    Return your response as a JSON object with these exact keys:
    {{"improved_question": "the improved question", "improved_options": ["option1", "option2", "option3", "option4"] if multiple choice, "improved_answer": "the improved answer if needed", "explanation": "brief explanation of improvements"}}
    """
    
    try:
        # Send to Gemini for improvement
        response = model.generate_content(improvement_prompt)
        
        # Extract JSON from response
        result_text = response.text
        
        # Try to find JSON in the response
        try:
            # First try to parse the whole response as JSON
            improvement_result = json.loads(result_text)
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from markdown code blocks
            import re
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', result_text)
            if json_match:
                try:
                    improvement_result = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    improvement_result = {}
            else:
                # Last resort: try to find anything that looks like JSON
                json_match = re.search(r'{[\s\S]*}', result_text)
                if json_match:
                    try:
                        improvement_result = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        improvement_result = {}
                else:
                    improvement_result = {}
        
        # Update question with improvements if available
        if "improved_question" in improvement_result:
            question['question'] = improvement_result["improved_question"]
            
        if "improved_answer" in improvement_result:
            question['answer'] = improvement_result["improved_answer"]
            
        if question_type == 'multiple_choice' and "improved_options" in improvement_result:
            if isinstance(improvement_result["improved_options"], list) and len(improvement_result["improved_options"]) >= 4:
                question['options'] = improvement_result["improved_options"]
        
        # Store improvement explanation
        if "explanation" in improvement_result:
            question['improvement_explanation'] = improvement_result["explanation"]
        
        return question
        
    except Exception as e:
        print(f"Error during Gemini improvement: {e}")
        return question

def iteratively_refine_question(question: Dict[str, Any], max_iterations: int = 3, quality_threshold: float = 0.85) -> Dict[str, Any]:
    """
    Iteratively refines a question until it meets a quality threshold or max iterations is reached
    
    Args:
        question: Dictionary containing question data
        max_iterations: Maximum number of improvement iterations
        quality_threshold: Minimum score to consider a question adequate (0.0-1.0)
        
    Returns:
        Refined question dictionary with history of improvements
    """
    # Initialize refinement history
    question['refinement_history'] = []
    current_question = question.copy()
    
    # Store original question
    original_question = question.copy()
    if 'gemini_validation' in original_question:
        del original_question['gemini_validation']
    if 'refinement_history' in original_question:
        del original_question['refinement_history']
    
    for iteration in range(max_iterations):
        # First validate the current question
        validated_question = validate_question(current_question)
        
        # Get the validation score
        validation_result = validated_question.get('gemini_validation', {})
        score = validation_result.get('overall_score', 0.0)
        
        # Store this iteration in history
        iteration_record = {
            'iteration': iteration,
            'score': score,
            'question': current_question.get('question', ''),
            'feedback': validation_result.get('feedback', '')
        }
        
        if question.get('type') == 'multiple_choice' and 'options' in current_question:
            iteration_record['options'] = current_question['options']
        
        question['refinement_history'].append(iteration_record)
        
        # Check if we've reached the quality threshold
        if score >= quality_threshold:
            print(f"Question reached quality threshold after {iteration+1} iterations (Score: {score:.2f})")
            # Update the original question with improved version
            for key, value in current_question.items():
                if key != 'refinement_history':
                    question[key] = value
            
            question['met_quality_threshold'] = True
            return question
        
        # If not at threshold, improve and continue
        print(f"Iteration {iteration+1}: Score {score:.2f} below threshold {quality_threshold}. Improving...")
        current_question = improve_question_with_gemini(current_question)
    
    # If we've exhausted iterations, return the best version
    print(f"Max iterations ({max_iterations}) reached. Using best version.")
    
    # Find the iteration with the highest score
    best_iteration = max(question['refinement_history'], key=lambda x: x.get('score', 0))
    best_score = best_iteration.get('score', 0)
    
    if best_score > validation_result.get('overall_score', 0):
        # Find the corresponding question state from history
        for i, record in enumerate(question['refinement_history']):
            if record is best_iteration:
                # Get the question from this iteration
                question['question'] = record.get('question', question['question'])
                if 'options' in record and question.get('type') == 'multiple_choice':
                    question['options'] = record['options']
                break
    else:
        # Keep the current version as it's the best
        for key, value in current_question.items():
            if key != 'refinement_history':
                question[key] = value
    
    question['met_quality_threshold'] = False
    return question

def categorize_and_validate(question: Dict[str, Any]) -> Dict[str, Any]:
    """
    Categorize the question and apply specialized validation based on its category
    
    Args:
        question: Dictionary containing question data
        
    Returns:
        Question with category-specific validation applied
    """
    model = get_model()
    if not model:
        return validate_question(question)  # Fall back to standard validation
    
    # Extract question text
    question_text = question.get('question', '')
    question_type = question.get('type', 'structured')
    
    # Determine educational category and cognitive level
    categorization_prompt = f"""
    Analyze this educational question and categorize it.
    
    Question: {question_text}
    
    Return a JSON object with these keys:
    {{
        "subject_area": "primary subject area (e.g. Computer Science, Mathematics)",
        "specific_topic": "specific topic within the subject area",
        "bloom_taxonomy_level": "one of: Remember, Understand, Apply, Analyze, Evaluate, Create",
        "question_complexity": "Low, Medium, High",
        "detailed_category": "detailed categorization like 'Algorithmic Thinking', 'Theory Application'"
    }}
    """
    
    try:
        response = model.generate_content(categorization_prompt)
        categorization = json.loads(response.text)
        
        # Add categorization to question
        question['educational_metadata'] = categorization
        
        # Now validate with category-specific criteria
        bloom_level = categorization.get('bloom_taxonomy_level', '').lower()
        
        # Adjust validation based on Bloom's taxonomy level
        if bloom_level in ['analyze', 'evaluate', 'create']:
            # For higher-order thinking questions, use iterative refinement
            return iteratively_refine_question(question, max_iterations=3, quality_threshold=0.8)
        else:
            # For knowledge-based questions, standard validation is sufficient
            return validate_question(question)
            
    except Exception as e:
        print(f"Error during question categorization: {e}")
        return validate_question(question)  # Fall back to standard validation

def generate_questions_with_gemini(context, num_questions=5, question_types=None, difficulty=None):
    """
    Generate questions using Gemini model with improved structure
    
    Args:
        context: Document content
        num_questions: Number of questions to generate
        question_types: Types of questions to generate
        difficulty: Question difficulty level
        
    Returns:
        List of question dictionaries
    """
    model = get_model()
    if not model:
        raise Exception("Model not available")
    
    if not question_types:
        question_types = ["multiple_choice", "short_answer"]
    
    # For very long contexts, trim to a reasonable length
    context_to_use = context[:15000] if len(context) > 15000 else context
    
    # Determine number of each question type to generate
    num_mc = num_questions
    num_structured = 0
    
    if "multiple_choice" in question_types and "short_answer" in question_types:
        num_mc = num_questions // 2
        num_structured = num_questions - num_mc
    elif "short_answer" in question_types and "multiple_choice" not in question_types:
        num_mc = 0
        num_structured = num_questions
    
    all_questions = []
    
    # Generate balanced questions with a single prompt to ensure better context understanding
    balanced_prompt = f"""
    Generate exactly {num_questions} high-quality questions based on the following educational content.
    
    CONTENT:
    ```
    {context_to_use}
    ```
    
    REQUIREMENTS:
    1. Each question must test understanding of specific concepts from the content
    2. Questions must be clearly worded and unambiguous
    3. All questions must be answerable from the provided content
    4. No duplicate or highly similar questions
    5. Questions should be of {difficulty if difficulty else "varying"} difficulty
    6. Generate {num_mc} multiple-choice questions and {num_structured} short-answer questions
    
    QUESTION FORMATS:
    
    For multiple-choice questions:
    - Each question must have exactly 4 options (labeled A, B, C, D)
    - Provide well-written distractors (wrong options) that are plausible but clearly incorrect
    - Each question must have exactly ONE correct answer
    - Include a brief explanation for why the correct answer is right
    
    For short-answer questions:
    - Questions should require a paragraph-length response
    - Provide a comprehensive model answer for each question
    - Ensure the question tests deeper understanding of the concepts
    
    FORMAT:
    Return ONLY a JSON array where each question object has these exact fields:
    
    For multiple-choice questions:
    - "question": A clear, complete question ending with a question mark
    - "options": An array of 4 strings representing options A, B, C, and D
    - "answer": A single letter (A, B, C, or D) indicating the correct answer
    - "explanation": Brief explanation for why the correct answer is right
    - "type": Always "multiple_choice"
    
    For short-answer questions:
    - "question": A clear, complete question ending with a question mark
    - "answer": A comprehensive model answer
    - "type": Always "structured"
    
    For example:
    [
      {{
        "question": "What is the primary purpose of confidentiality in computer security?",
        "options": [
          "To ensure data is not modified by unauthorized users",
          "To prevent unauthorized access to sensitive information",
          "To verify the identity of users accessing the system",
          "To ensure system availability during disruptions"
        ],
        "answer": "B",
        "explanation": "Confidentiality is concerned with preventing unauthorized access to sensitive information, keeping private data secure from those who should not see it.",
        "type": "multiple_choice"
      }},
      {{
        "question": "Explain how the principle of least privilege contributes to system security.",
        "answer": "The principle of least privilege enhances system security by ensuring users and processes have only the minimum permissions necessary to perform their functions. This limits the potential damage from accidents, errors, or malicious attacks by restricting access rights. If a user account or process is compromised, the attacker only gains limited access rather than full system privileges. This principle reduces the attack surface and minimizes the risk of privilege escalation attacks.",
        "type": "structured"
      }}
    ]

    Return ONLY the JSON array with no additional text or explanation.
    """
    
    try:
        response = model.generate_content(balanced_prompt, generation_config={
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        })
        
        # Extract and parse the JSON response
        import json
        import re
        
        # Try to find JSON array in the response
        json_match = re.search(r'\[\s*{.*}\s*\]', response.text.replace('\n', ' '), re.DOTALL)
        if json_match:
            questions = json.loads(json_match.group(0))
        else:
            # Try direct parsing
            questions = json.loads(response.text)
        
        # Process and validate each question
        for q in questions:
            if isinstance(q, dict) and "question" in q and "type" in q:
                # Process based on question type
                if q["type"] == "multiple_choice" and "options" in q and "answer" in q:
                    # Ensure multiple choice questions have all required fields
                    q["difficulty"] = difficulty or "medium"
                    q["context"] = context_to_use[:300] + "..." if len(context_to_use) > 300 else context_to_use
                    q["source"] = "gemini"
                    all_questions.append(q)
                elif q["type"] == "structured" and "answer" in q:
                    # Ensure structured questions have all required fields
                    q["difficulty"] = difficulty or "medium"
                    q["context"] = context_to_use[:300] + "..." if len(context_to_use) > 300 else context_to_use
                    q["source"] = "gemini"
                    all_questions.append(q)
    
    except Exception as e:
        print(f"Error generating questions: {e}")
    
    return all_questions