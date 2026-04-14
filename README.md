# 🎬 Movie Recommendation System

A simple, modern **Movie Recommendation System** built with **Python** and **Streamlit** that uses **Content-Based Filtering** with **Cosine Similarity** to suggest movies based on your preferences.

---

## 📌 Features

- Enter **3 favourite movies** and get **5 personalized recommendations**
- Uses **TF-IDF Vectorization** and **Cosine Similarity** for content-based filtering
- Combines **genre, keywords, director, and rating** for accurate recommendations
- Clean, modern UI with a dark gradient theme
- Input validation and error handling
- Beginner-friendly, well-commented code

---

## 🗂 Project Structure

```
├── app.py              # Streamlit web app (UI + interaction)
├── recommender.py      # Recommendation engine (TF-IDF + Cosine Similarity)
├── movies.csv          # Movie dataset (50 popular movies)
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/NadafAyan/Quantum-Inspired-Movie-Recommendation-System-.git
cd Quantum-Inspired-Movie-Recommendation-System-
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🧠 How It Works

### Content-Based Filtering

This system uses **content-based filtering**, which recommends items similar to what the user already likes based on item attributes.

### Algorithm Steps

1. **Feature Combination** — For each movie, we combine `genre`, `keywords`, `director`, and `rating` into a single text string.
2. **TF-IDF Vectorization** — We convert the combined text into numerical vectors using **Term Frequency – Inverse Document Frequency (TF-IDF)**. This measures the importance of each word.
3. **Cosine Similarity** — We calculate the cosine of the angle between any two movie vectors. A value of 1 means identical, and 0 means completely different.
4. **Averaging Scores** — For the 3 input movies, we average their similarity scores with all other movies, then pick the top 5.

### Key Formula

```
Cosine Similarity = (A · B) / (||A|| × ||B||)
```

Where A and B are TF-IDF vectors of two movies.

---

## 📊 Dataset

The dataset (`movies.csv`) contains **50 popular movies** with the following columns:

| Column       | Description                          |
|-------------|--------------------------------------|
| `movie_name` | Name of the movie                    |
| `genre`      | Genre(s) of the movie                |
| `director`   | Director of the movie                |
| `year`       | Release year                         |
| `rating`     | IMDb rating (out of 10)              |
| `keywords`   | Keywords describing the movie plot   |

---

## 🛠 Tech Stack

| Technology     | Purpose                                |
|----------------|----------------------------------------|
| Python         | Programming language                   |
| Streamlit      | Web app framework                      |
| pandas         | Data manipulation                      |
| scikit-learn   | TF-IDF vectorization & cosine similarity |

---

## 📸 Screenshot

After running the app, you'll see:
1. A title and subtitle
2. Three input fields for your favourite movies
3. A "Recommend Movies" button
4. A list of 5 recommended movies with details

---

## 🙋 Viva Questions & Answers

**Q: What is content-based filtering?**
A: It recommends items based on their features/content. It finds items similar to what the user already likes.

**Q: What is TF-IDF?**
A: Term Frequency–Inverse Document Frequency. It converts text to numbers by measuring how important each word is in a document relative to the entire collection.

**Q: What is cosine similarity?**
A: It measures the cosine of the angle between two vectors. It tells us how similar two movies are based on their features. Range: 0 (different) to 1 (identical).

**Q: Why combine features?**
A: Combining genre, keywords, director, and rating gives a richer representation of each movie, leading to more accurate recommendations.

**Q: What library is used for the web app?**
A: Streamlit — a Python library that makes it easy to create web apps without HTML/CSS/JS knowledge.

---

## 📄 License

This project is licensed under the MIT License.