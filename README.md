### Research overview based to this Project

### 🔍 **System Overview**

The **Recommendation System for Identification of Diabetes Risk and Providing Health Guidelines** is a modular, AI-driven health platform designed to address the rising concern of **Type 2 diabetes among Sri Lankan office workers aged 20–50**. The system integrates advanced **machine learning, natural language processing**, and **graph-based AI models** to deliver **personalized health recommendations** in a culturally contextualized and holistic manner.

The solution is structured around **four key modules**, each addressing a critical aspect of diabetes management:

---

### 🧠 **1. Diabetic Risk Identification Module**

* **Purpose**: Predict the user's risk of developing Type 2 diabetes.
* **Method**: Uses a **Linear Regression** model that considers variables like age, BMI, waist circumference, family history, blood pressure, diet, and symptoms (e.g., thirst, fatigue, frequent urination).
* **Outcome**: Produces a **diabetes risk score** which drives all subsequent module decisions.

---

### 🍱 **2. Nutrition Risk Assessment & Meal Plan Recommendation Module**

* **Purpose**: Assess nutritional risk and recommend diabetes-friendly meals.
* **Modeling**:

  * **Random Forest Regression** is used to compute nutritional risk.
  * A **Nutrition-Related Knowledge Graph (NRKG)** built using **Graph Neural Networks (GNNs)** generates personalized meal plans.
* **Input Factors**: Macronutrient intake, hydration, meal regularity, sugar consumption, etc.
* **Output**: Weekly meal plans filtered by disease conditions and dietary preferences.

---

### 🏋️ **3. Physical Activity Risk & Workout Recommendation Module**

* **Purpose**: Evaluate physical inactivity risks and recommend suitable workouts.
* **Methodologies**:

  * **Decision Tree classifier** for risk identification.
  * **Natural Language Processing (SBERT)** to parse user feedback.
  * **Isolation Forest** for filtering unsafe or anomalous exercises.
* **Output**: Customized, safe, and goal-oriented exercise routines considering time constraints, physical limitations, and health goals.

---

### 😟 **4. Mental Health Stress Detection & Recommendation Module**

* **Purpose**: Detect stress levels and recommend stress-relief techniques.
* **Techniques Used**:

  * **Facial Emotion Recognition** using CNNs.
  * **Questionnaire-based ML analysis** of user-reported stress symptoms.
  * **CBT-inspired recommendation system** for stress management.
* **Validation**: Outputs are cross-verified by psychiatrists by correlating facial expression results with questionnaire responses.

---

### ⚙️ **Technical Stack & Integration**

* **Languages & Frameworks**: Python, JavaScript, React/Next.js, SQL
* **AI/ML Libraries**: TensorFlow, PyTorch, Scikit-learn, PyTorch Geometric, OpenCV, Sentence Transformers
* **Tools**: Google Colab, Jupyter, VS Code, GitHub
* **Data Handling**: Pandas, NumPy, Matplotlib

---

### 🎯 **Key Strengths**

* **Personalization**: Tailors recommendations to individual health profiles.
* **Multi-domain Integration**: Combines nutrition, physical activity, mental health, and risk detection.
* **Cultural Relevance**: Accounts for local dietary practices and work culture in Sri Lanka.
* **Explainability & Interactivity**: Models are interpretable and interactive, supporting user prompts and preferences.

---

### 📈 **Impact**

By providing holistic, AI-powered health guidance, this system empowers diabetic office workers to take control of their health and reduce the long-term burden of diabetes. It also serves as a template for future healthcare AI systems in culturally similar contexts.


