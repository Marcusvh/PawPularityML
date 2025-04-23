from tkinter import messagebox

def discussion_precision_vs_recall():
    # Discussion about the models
    discussion_text = (
        "Precision vs Recall\n"
        "Precision er forholdet mellem sande positive forudsigelser og det samlede antal forudsagte positive.\n"
        "Formel for Precision: Precision = TP / (TP + FP)\n\n"
        "Recall er forholdet mellem sande positive forudsigelser og det samlede antal faktiske positive.\n"
        "Formel for Recall: Recall = TP / (TP + FN)\n\n"
        "Hvornår man bør prioritere Precision frem for Recall:\n"
        "Hvis omkostningen ved en falsk positiv er høj, bør man prioritere Precision. For eksempel i spamfiltrering: at markere en legitim e-mail som spam (falsk positiv) kan være til stor gene for brugeren.\n\n"
        "Hvornår man bør prioritere Recall frem for Precision:\n"
        "Hvis omkostningen ved en falsk negativ er høj, bør man prioritere Recall. For eksempel i medicinske diagnoser: at undlade at opdage en sygdom (falsk negativ) kan have alvorlige konsekvenser for patienten."
    )
    messagebox.showinfo("Discussion Precision vs Recall", discussion_text)

def discussion_bias_variance_trees():
    # Discussion about the models
    discussion_text = (
        "Bias vs Variance\n\n"
        "Bias refererer til fejlene der opstår, når en model antager for meget om dataene. Lav bias er godt\n"
        "Lav bias betyder at den antager mindre og derfor kan bedre tilpasse sig dataene. Den kan derfor også være tæt på trænings data\n"
        "Høj bias betyder at den antager mere når den træner og derfor passer ikke sig ikke ind til trænings dataene. Hvilket fører til underfitting.\n\n"
        "Variance er hvor god modellen er til at tilpasse sig til ny data under det samme dataset. Den måler sprædningen fra dataens mean (gennemsnit). Lav variance er godt\n"
        "Lav variance betyder at den ikke er så følsom overfor ændringer i dataene og derfor kan bedre tilpasse sig til nye data. Dog kan lav variance også være tegn på underfitting, hvis modellen er for simpel og dermed fejler med at se tendenser.\n"
        "Høj variance betyder at den er meget følsom overfor ændringer i dataene og derfor kan overfitte til trænings dataene. Den har tilpasset sig for meget til sin trænings data, at den fejler på ny trænings data.\n\n"
        )
    messagebox.showinfo("Discussion Bias vs Variance", discussion_text)