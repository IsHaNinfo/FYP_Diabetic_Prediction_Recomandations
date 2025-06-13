from src.components.Exersices_Recommendations.exercise_recommander import recommend

def format_paragraphs(recommendations):
    """Format the exercise recommendations into readable paragraphs."""
    if not recommendations:
        return "No recommendations available."
    
    formatted_text = []
    
    for routine_block in recommendations.get("routine", []):
        role = routine_block.get("role", "")
        exercises = routine_block.get("exercises", [])
        
        formatted_text.append(f"\n{role.upper()} EXERCISES:")
        
        for exercise in exercises:
            formatted_text.append(f"\n{exercise['name']}")
            formatted_text.append(f"Level: {exercise['level']}")
            formatted_text.append(f"Equipment: {exercise['equipment']}")
            formatted_text.append(f"Sets and Reps: {exercise['reps_sets']}")
            
            if exercise.get('instructions'):
                formatted_text.append("Instructions:")
                for i, step in enumerate(exercise['instructions'], 1):
                    formatted_text.append(f"{i}. {step}")
            
            formatted_text.append("")  # Add spacing between exercises
    
    return "\n".join(formatted_text)
