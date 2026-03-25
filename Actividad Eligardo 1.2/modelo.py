import pandas as pd
import numpy as np
import random
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ==========================================
# 1. ADVANCED TEXT NORMALIZATION
# ==========================================

LEET_MAP = str.maketrans({
    '0': 'o', '1': 'i', '2': 'z', '3': 'e', '4': 'a', 
    '5': 's', '6': 'g', '7': 't', '8': 'b', '9': 'g',
    '@': 'a', '!': 'i', '$': 's', '+': 't', '|': 'i',
    '(': 'c', '[': 'c', '{': 'c', '<': 'c'
})

WORD_EXPANSIONS = {
    'u': 'you', 'r': 'are', 'ur': 'your', 'yu': 'you', 'ar': 'are',
    'tu': 'to', 'plz': 'please', 'pls': 'please', 'k': 'ok', 'y': 'why',
    'b4': 'before', 'gr8': 'great', 'stfu': 'shut the fuck up',
    'gtfo': 'get the fuck out', 'idk': 'i do not know', 'gg': 'good game',
    'wp': 'well played', 'pai': 'pay', 'tovch': 'touch'
}

def normalize_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = text.translate(LEET_MAP)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    words = text.split()
    new_words = []
    i = 0
    while i < len(words):
        j = i
        while j < len(words) and len(words[j]) == 1: j += 1
        if (j - i) > 1:
            current_seq = "".join(words[i:j])
            if current_seq == "ur":
                new_words.extend(["you", "are"])
            else:
                new_words.append(current_seq)
            i = j
        else:
            new_words.append(words[i])
            i += 1
    
    text = " ".join(new_words)
    words = text.split()
    words = [WORD_EXPANSIONS.get(w, w) for w in words]
    return re.sub(r'\s+', ' ', " ".join(words)).strip()

# ==========================================
# 2. DATA GENERATION
# ==========================================

def create_dataset():
    toxic_words = [
        "idiot", "loser", "trash", "garbage", "noob", "bitch", "asshole", 
        "retard", "faggot", "suck", "useless", "feeder", "thrower", "manco", 
        "pendejo", "idiota", "basura", "hdp", "clown", "bot"
    ]
    toxic_templates = [
        "you are a {}", "{} teammate", "stop being a {}", "you play like {}", 
        "get out of here {}", "fuck you {}", "uninstall the game {}", 
        "you're so {}", "{} player", "{} noob", "go cry {}", "useless {}"
    ]
    indirect_toxic = [
        "lick my wood", "go touch grass", "uninstall the game", "you're useless",
        "worst teammate ever", "my grandma plays better than you", "go play tetris",
        "nice aim bot", "how much did you pay for that account", "wood division player",
        "please stop typing and play", "report this guy", "throwing on purpose",
        "waste of space", "your brain is lagging", "nice throw", "stay in bronze",
        "vete a dormir", "asco de equipo", "no sirves para nada", "reporten al jg",
        "manco de mierda", "eres malisimo", "deja de fedear", "unins7all pls"
    ]
    
    non_toxic = [
        "nice shot", "good game", "well played", "let's win this", "gg wp",
        "gl hf", "nice try", "maybe next time", "can someone help me?",
        "i need heals", "go for the objective", "defending B", "enemy spotted",
        "thanks for the carry", "no problem", "sorry my bad", "wait for me",
        "let's group up", "good job team", "i'm lagging a bit", "brb",
        "haha lol", "omg that was close", "wp everyone", "love this game",
        "great teamwork", "buena partida", "bien jugado", "vamos a ganar",
        "ayuda en dragon", "necesito cura", "buen disparo", "gracias",
        "perdón, fue mi error", "esperenme", "juguemos juntos", "buen trabajo",
        "hi", "hello", "how are you", "ready?", "go go go", "yes", "no",
        "wait", "coming", "on my way", "i'll help", "good luck", "have fun",
        "that was amazing", "wow", "unbelievable", "close one", "solid play",
        "nice pass", "great defense", "keep it up", "we can do this", "don't give up",
        "yeah", "yep", "sure", "of course", "okay", "alright", "no worries"
    ]

    data = []
    # Generate Toxic (Label 1)
    for _ in range(400):
        word = random.choice(toxic_words)
        if random.random() > 0.4:
            # Add some manual leet speak for training diversity
            word = word.replace('e', '3').replace('a', '4').replace('i', '1')
        data.append([random.choice(toxic_templates).format(word), 1])
    for _ in range(150):
        data.append([random.choice(indirect_toxic), 1])
        
    # Generate Non-Toxic (Label 0) - Increased volume and variety
    for _ in range(700):
        data.append([random.choice(non_toxic), 0])
    
    # Add random casing and typos
    for i in range(len(data)):
        text = data[i][0]
        if random.random() > 0.7: text = text.upper()
        if random.random() > 0.8 and len(text) > 5:
            idx = random.randint(0, len(text)-2)
            text = text[:idx] + text[idx+1] + text[idx] + text[idx+2:]
        data[i][0] = text

    df = pd.DataFrame(data, columns=["utterance", "label"])
    return df.drop_duplicates().sample(frac=1).reset_index(drop=True)

# ==========================================
# 3. TRAINING & EVALUATION
# ==========================================

def train_model(df):
    df['clean_utterance'] = df['utterance'].apply(normalize_text)
    
    # Using a mix of word and char n-grams to improve precision
    model = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 5), 
            analyzer='char_wb', 
            max_features=15000,
            min_df=2
        )),
        ('clf', LogisticRegression(C=5, solver='liblinear', class_weight='balanced'))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_utterance'], df['label'], test_size=0.2, random_state=42
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred))
    return model

if __name__ == "__main__":
    print("Generating and normalizing dataset...")
    df = create_dataset()
    df.to_csv("toxic_dataset.csv", index=False)
    
    print(f"Dataset ready: {len(df)} rows.")
    model = train_model(df)
    
    print("\n" + "="*40)
    print("TOXIC DETECTOR - PRECISION MODE")
    print("Type 'exit' to quit.")
    print("="*40)
    
    while True:
        user_input = input("\nTest phrase: ")
        if user_input.lower() == 'exit': break
        
        normalized = normalize_text(user_input)
        prediction = model.predict([normalized])[0]
        prob = model.predict_proba([normalized])[0]
        
        result = "TOXIC" if prediction == 1 else "NON-TOXIC"
        confidence = prob[1] if prediction == 1 else prob[0]
        
        # Heuristic: if confidence is low, default to non-toxic to reduce false positives
        if prediction == 1 and confidence < 0.65:
            result = "NON-TOXIC (Low confidence)"
            
        print(f"Prediction: {result} (Confidence: {confidence:.2%})")
        print(f"Normalized as: '{normalized}'")
