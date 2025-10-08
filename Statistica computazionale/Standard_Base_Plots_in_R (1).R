# Standard Base Plots in R
# ========================
#
# Questo script contiene tutti i comandi R estratti dal notebook
# "Standard_Base_Plots_in_R.ipynb" con commenti esplicativi.
# Gli esempi coprono i grafici base R più comuni per l'esplorazione dei dati
# e introduzioni a dplyr e ggplot2.

# ==============================================================================
# PARTE 1: GRAFICI BASE R
# ==============================================================================

# Generazione dei dati di esempio per i grafici base
# --------------------------------------------------
# Creiamo un vettore numerico con distribuzione normale
data <- rnorm(100, mean = 10, sd = 2)

# 1. ISTOGRAMMA
# -------------
# L'istogramma mostra la distribuzione di una variabile numerica
# dividendo i dati in intervalli (bins) e contando le osservazioni in ciascuno
hist(data, main = "Histogram of data", xlab = "Value", col = "lightblue", border = "white")

# 2. BOXPLOT
# ----------
# Il boxplot riassume la distribuzione mostrando mediana, quartili e outlier
boxplot(data, main = "Boxplot of data", col = "orange")

# 3. SCATTERPLOT
# --------------
# Il grafico a dispersione mostra la relazione tra due variabili numeriche
set.seed(42)
data2 <- data + rnorm(100) # Seconda variabile correlata alla prima

# Scatterplot con personalizzazione dei punti
plot(data, data2, main = "Scatterplot of data vs data2", xlab = "data", ylab = "data2", col = "blue", pch = 19)

# 4. BOXPLOT PER GRUPPI
# ---------------------
# Confronto della distribuzione di una variabile numerica tra diverse categorie
group <- sample(c("A", "B"), 100, replace = TRUE) # Variabile categorica

# Boxplot raggruppato per confrontare i gruppi
boxplot(data ~ group, main = "Boxplot of data by group", col = c("lightgreen", "lightpink"), xlab = "Group", ylab = "data")

# 5. BARPLOT
# ----------
# Il grafico a barre mostra la frequenza di ciascuna categoria
category_counts <- table(group) # Conta le frequenze per categoria
barplot(category_counts, main = "Barplot of group counts", col = c("lightgreen", "lightpink"), ylab = "Frequency")

# 6. GRAFICO A TORTA
# ------------------
# Il grafico a torta mostra le proporzioni delle categorie
pie(category_counts, main = "Pie chart of group proportions", col = c("lightgreen", "lightpink"))

# ==============================================================================
# PARTE 2: INTRODUZIONE A DPLYR
# ==============================================================================

# Installazione e caricamento di dplyr se necessario
if (!require(dplyr)) install.packages("dplyr")
library(dplyr)

# Creazione di un data frame di esempio per dplyr
df <- data.frame(
  x = rnorm(100, mean = 10, sd = 2),
  y = rnorm(100, mean = 20, sd = 5),
  group = sample(c("A", "B"), 100, replace = TRUE)
)

# WORKFLOW DPLYR CON PIPE OPERATOR
# --------------------------------
# dplyr utilizza il pipe operator %>% per concatenare operazioni
# Ogni funzione trasforma i dati e li passa alla successiva

df_summary <- df %>%
  filter(x > 10) %>% # Mantieni solo le righe dove x > 10
  mutate(z = x + y) %>% # Aggiungi una nuova colonna z (somma di x e y)
  group_by(group) %>% # Raggruppa per la variabile 'group'
  summarise( # Calcola statistiche riassuntive per gruppo
    mean_x = mean(x),
    mean_y = mean(y),
    mean_z = mean(z),
    n = n() # Conta il numero di osservazioni
  ) %>%
  arrange(desc(mean_z)) # Ordina per mean_z in modo decrescente

# Mostra il risultato
df_summary

# ==============================================================================
# PARTE 3: INTRODUZIONE A GGPLOT2
# ==============================================================================

# Installazione e caricamento di ggplot2 se necessario
if (!require(ggplot2)) install.packages("ggplot2")
library(ggplot2)

# ESEMPIO COMPLETO: ANATOMIA DI UN GRAFICO GGPLOT2
# -------------------------------------------------
# Questo esempio mostra tutti i componenti principali di ggplot2
ggplot(df, aes(x = x, y = y, color = group)) + # Data e aesthetics
  geom_point(size = 2) + # Geometria: punti
  stat_smooth(method = "lm", se = FALSE) + # Statistica: linea di regressione
  scale_color_brewer(palette = "Set1") + # Scala colori
  facet_wrap(~group) + # Facet: pannelli separati per gruppo
  coord_cartesian(xlim = c(5, 15)) + # Coordinate: limiti assi
  labs( # Labels: titoli e etichette
    title = "Grammar of Graphics Example",
    x = "X Value",
    y = "Y Value",
    color = "Group"
  ) +
  theme_minimal() # Tema: aspetto generale

# ESEMPI DI GRAFICI GGPLOT2
# -------------------------

# 1. Scatterplot base
ggplot(df, aes(x = x, y = y)) +
  geom_point() +
  labs(title = "Basic ggplot2 Scatterplot", x = "x", y = "y")

# 2. Scatterplot colorato per gruppo
ggplot(df, aes(x = x, y = y, color = group)) +
  geom_point(size = 2) +
  labs(title = "Scatterplot Colored by Group")

# 3. Istogramma con ggplot2
ggplot(df, aes(x = x)) +
  geom_histogram(bins = 20, fill = "skyblue", color = "white") +
  labs(title = "Histogram with ggplot2", x = "x")

# 4. Boxplot per gruppo
ggplot(df, aes(x = group, y = y, fill = group)) +
  geom_boxplot() +
  labs(title = "Boxplot by Group", x = "Group", y = "y")

# ESEMPI AVANZATI DI GGPLOT2
# --------------------------

# 5. Scatterplot personalizzato con linea di regressione
ggplot(df, aes(x = x, y = y, color = group)) +
  geom_point(size = 3, alpha = 0.7) +
  labs(title = "Customized Scatterplot", x = "x", y = "y", color = "Group") +
  theme_minimal()

# Aggiunta di linea di regressione
ggplot(df, aes(x = x, y = y)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  labs(title = "Scatterplot with Regression Line")

# 6. Grafici con facet (pannelli multipli)
ggplot(df, aes(x = x, y = y)) +
  geom_point(color = "steelblue") +
  facet_wrap(~group) +
  labs(title = "Faceted Scatterplots by Group")

# 7. Barplot con ggplot2
ggplot(df, aes(x = group, fill = group)) +
  geom_bar() +
  labs(title = "Barplot of Group Counts", x = "Group", y = "Count")

# Barplot raggruppato con una seconda variabile categorica
df$cat <- sample(c("X", "Y"), 100, replace = TRUE)

ggplot(df, aes(x = group, fill = cat)) +
  geom_bar(position = "dodge") +
  labs(title = "Grouped Barplot", fill = "Category")

# 8. Grafico di densità
ggplot(df, aes(x = x, fill = group)) +
  geom_density(alpha = 0.5) +
  labs(title = "Density Plot by Group", x = "x")

# 9. Boxplot con punti sovrapposti (jittered)
ggplot(df, aes(x = group, y = y, fill = group)) +
  geom_boxplot(outlier.shape = NA) + # Boxplot senza outlier
  geom_jitter(width = 0.2, alpha = 0.5, color = "black") + # Punti sparsi
  labs(title = "Boxplot with Jittered Points", x = "Group", y = "y")

# SALVATAGGIO DEI GRAFICI
# -----------------------

# Salva l'ultimo grafico come file PNG
ggsave("boxplot_jittered.png", width = 6, height = 4)

# Salva un grafico specifico assegnandolo a una variabile
p <- ggplot(df, aes(x = x, y = y)) +
  geom_point()
ggsave("scatterplot.png", plot = p, width = 5, height = 4)

# ==============================================================================
# FINE DELLO SCRIPT
# ==============================================================================
#
# Questo script fornisce una panoramica completa dei grafici base R e
# un'introduzione a dplyr e ggplot2. Ogni sezione è commentata per
# facilitare l'apprendimento e la comprensione dei concetti.
#
# Per ulteriori informazioni:
# - Documentazione R base: help(plot), help(hist), etc.
# - dplyr: https://dplyr.tidyverse.org/
# - ggplot2: https://ggplot2.tidyverse.org/
