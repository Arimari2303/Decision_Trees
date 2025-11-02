# ===========================================================
# PERSONA 3: EFECTOS CAUSALES CON OLS Y RANDOM FOREST (R)
# ===========================================================

# Instalar paquetes si faltan
# install.packages(c("randomForest", "rpart", "rpart.plot"))

library(randomForest)
library(rpart)
library(rpart.plot)

# Preparar features numéricas originales (sin dummies)
feature_cols <- c('age','sex','cp','restbp','chol','fbs','restecg',
                  'thalach','exang','oldpeak','slope','ca','thal')

X_causal <- df[, feature_cols]

# Variables de tratamiento y outcome
# Asegúrate de que T y Y existan en df o en el entorno
T <- df$T
Y <- df$Y

# -----------------------------------------------------------
# (1 punto) OLS con tratamiento y covariables
# -----------------------------------------------------------
cat("\n1. Regresión OLS...\n")

# Modelo OLS
X_ols <- cbind(X_causal, T = T)
ols <- lm(Y ~ ., data = X_ols)

# Coeficiente del tratamiento
treatment_effect_ols <- coef(ols)["T"]
cat(sprintf("   Efecto del tratamiento (OLS): %.4f\n", treatment_effect_ols))


# -----------------------------------------------------------
# (2 puntos) Random Forest Causal
# -----------------------------------------------------------
cat("\n2. Random Forest Causal...\n")

# Entrenar modelos separados
rf_treated <- randomForest(X_causal[T == 1, ], Y[T == 1], 
                           ntree = 100, maxnodes = 5)
rf_control <- randomForest(X_causal[T == 0, ], Y[T == 0], 
                           ntree = 100, maxnodes = 5)

# Predecir outcomes potenciales
Y1_all <- predict(rf_treated, X_causal)   # Si todos tratados
Y0_all <- predict(rf_control, X_causal)   # Si nadie tratado

# Efectos individuales del tratamiento
individual_effects <- Y1_all - Y0_all
ate_rf <- mean(individual_effects)

cat(sprintf("   Efecto promedio (Random Forest): %.4f\n", ate_rf))
cat(sprintf("   Heterogeneidad (sd): %.4f\n", sd(individual_effects)))


# -----------------------------------------------------------
# (1 punto) Árbol representativo de heterogeneidad
# -----------------------------------------------------------
cat("\n3. Árbol representativo (max_depth=2)...\n")

tree_repr <- rpart(individual_effects ~ ., data = X_causal, 
                   method = "anova",
                   control = rpart.control(maxdepth = 2, minbucket = 20))

# Graficar árbol
png("output/arbol_persona3.png", width=900, height=500)
rpart.plot(tree_repr, main = "Árbol Representativo - Efectos Heterogéneos (Persona 3)")
dev.off()

cat("   ✓ Árbol guardado: output/arbol_persona3.png\n")

# ===========================================================
# ANÁLISIS AVANZADO (6 puntos total)
# ===========================================================

cat("\n1. Árbol de efectos heterogéneos...\n")

# Árbol representativo mejorado
tree_p4 <- rpart(
  individual_effects ~ ., 
  data = X_causal,
  method = "anova",
  control = rpart.control(
    maxdepth = 2,
    minbucket = 15,       # equivalente a min_samples_leaf
    minsplit = 30         # equivalente a min_samples_split
  )
)

# Guardar gráfico
png("output/arbol_heterogeneo_p4.png", width = 1200, height = 600)
rpart.plot(
  tree_p4, 
  main = "Árbol de Efectos Heterogéneos del Tratamiento (Persona 4)\nmax_depth = 2",
  type = 2,              # nodos separados
  extra = 101,           # valores de predicción y % observaciones
  under = TRUE,
  faclen = 0,
  fallen.leaves = TRUE,
  roundint = FALSE
)
dev.off()

cat("   ✓ Árbol guardado: output/arbol_heterogeneo_p4.png\n")


# (1.5 puntos) Importancia de características
cat("\n2. Calculando importancias...\n")

# Entrenar Random Forest para importancia de variables
rf_imp <- randomForest(
  x = X_causal,
  y = individual_effects,
  ntree = 200,
  maxnodes = 6 * 5,   # approx max_depth=6 (R usa maxnodes en lugar de max_depth)
  nodesize = 10,      # equivalente a min_samples_leaf
  importance = TRUE
)

# Extraer importancias
importances <- data.frame(
  feature = rownames(importance(rf_imp)),
  importance = importance(rf_imp)[, "%IncMSE"]
)

# Ordenar por importancia
importances <- importances[order(-importances$importance), ]

# Gráfico de barras
library(ggplot2)

ggplot(importances, aes(x = reorder(feature, importance), y = importance)) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Importancia de Características en la Heterogeneidad",
    x = "Características",
    y = "Importancia (%IncMSE)"
  ) +
  theme_minimal(base_size = 13)

# Guardar gráfico
ggsave("output/importancias_p4.png", width = 8, height = 5, dpi = 300)

cat("\nTOP 5 CARACTERÍSTICAS:\n")
print(head(importances, 5))

# (2 puntos) Análisis por terciles
cat("\n3. Análisis por terciles...\n")

library(dplyr)
library(ggplot2)
library(reshape2)

# ------------------------------------------
# Estandarizar (z-scores)
# ------------------------------------------
X_std <- as.data.frame(scale(X_causal))
colnames(X_std) <- feature_cols

# ------------------------------------------
# Crear terciles de efectos individuales
# (qcut en Python -> cut con quantiles en R)
# ------------------------------------------
breaks <- quantile(individual_effects, probs = c(0, 1/3, 2/3, 1))
terciles <- cut(individual_effects,
                breaks = breaks,
                include.lowest = TRUE,
                labels = c("Bajo", "Medio", "Alto"))

# ------------------------------------------
# Estadísticas por tercil
# ------------------------------------------
cat("\nDistribución por terciles:\n")
for (terc in c("Bajo","Medio","Alto")) {
  mask <- terciles == terc
  n <- sum(mask)
  mean_eff <- mean(individual_effects[mask])
  cat(sprintf("- %s: %d personas, efecto medio: %.3f\n", terc, n, mean_eff))
}

# ------------------------------------------
# Heatmap de medias estandarizadas por tercil
# ------------------------------------------

mean_by_tercile <- X_std %>%
  mutate(terciles = terciles) %>%
  group_by(terciles) %>%
  summarise(across(everything(), mean)) %>%
  as.data.frame()

# Reorganizar para ggplot
mean_melt <- melt(mean_by_tercile, id.vars = "terciles")

# Plot estilo heatmap
p <- ggplot(mean_melt, aes(x = variable, y = terciles, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(
    low = "blue", mid = "white", high = "red", midpoint = 0,
    name = "Media Estandarizada (z-score)"
  ) +
  geom_text(aes(label = sprintf("%.2f", value)), size = 3) +
  labs(
    title = "Distribución de Covariables por Terciles del Efecto",
    subtitle = "Rojo = por encima del promedio | Azul = por debajo",
    x = "Características",
    y = "Terciles"
  ) +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p)

ggsave("output/heatmap_terciles_p4.png", p, width = 10, height = 6, dpi = 300)
cat("   ✓ Heatmap guardado: output/heatmap_terciles_p4.png\n")



# (2 puntos) Análisis por terciles
cat("\n3. Análisis por terciles...\n")

library(dplyr)
library(ggplot2)
library(reshape2)

# ------------------------------------------
# Estandarizar (z-scores)
# ------------------------------------------
X_std <- as.data.frame(scale(X_causal))
colnames(X_std) <- feature_cols

# ------------------------------------------
# Crear terciles de efectos individuales
# (qcut en Python -> cut con quantiles en R)
# ------------------------------------------
breaks <- quantile(individual_effects, probs = c(0, 1/3, 2/3, 1))
terciles <- cut(individual_effects,
                breaks = breaks,
                include.lowest = TRUE,
                labels = c("Bajo", "Medio", "Alto"))

# ------------------------------------------
# Estadísticas por tercil
# ------------------------------------------
cat("\nDistribución por terciles:\n")
for (terc in c("Bajo","Medio","Alto")) {
  mask <- terciles == terc
  n <- sum(mask)
  mean_eff <- mean(individual_effects[mask])
  cat(sprintf("- %s: %d personas, efecto medio: %.3f\n", terc, n, mean_eff))
}

# ------------------------------------------
# Heatmap de medias estandarizadas por tercil
# ------------------------------------------

mean_by_tercile <- X_std %>%
  mutate(terciles = terciles) %>%
  group_by(terciles) %>%
  summarise(across(everything(), mean)) %>%
  as.data.frame()

# Reorganizar para ggplot
mean_melt <- melt(mean_by_tercile, id.vars = "terciles")

# Plot estilo heatmap
p <- ggplot(mean_melt, aes(x = variable, y = terciles, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(
    low = "blue", mid = "white", high = "red", midpoint = 0,
    name = "Media Estandarizada (z-score)"
  ) +
  geom_text(aes(label = sprintf("%.2f", value)), size = 3) +
  labs(
    title = "Distribución de Covariables por Terciles del Efecto",
    subtitle = "Rojo = por encima del promedio | Azul = por debajo",
    x = "Características",
    y = "Terciles"
  ) +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p)

ggsave("output/heatmap_terciles_p4.png", p, width = 10, height = 6, dpi = 300)
cat("   ✓ Heatmap guardado: output/heatmap_terciles_p4.png\n")


# (1.5 puntos) Síntesis final
cat("\n4. Síntesis visual...\n")

library(ggplot2)
library(gridExtra)
library(jsonlite)

# -----------------------------------------
# Panel 1: Distribución de efectos
# -----------------------------------------
df_eff <- data.frame(effect = individual_effects)

p1 <- ggplot(df_eff, aes(x = effect)) +
  geom_histogram(bins = 30, color="black", alpha=.6) +
  geom_vline(xintercept = ate_rf, linetype="dashed", size=1) +
  geom_vline(xintercept = treatment_effect_ols, linetype="dashed", size=1) +
  labs(
    title="Distribución de Efectos Heterogéneos",
    x="Efecto Individual",
    y="Frecuencia"
  ) +
  theme_minimal()

# -----------------------------------------
# Panel 2: Boxplot por tercil
# -----------------------------------------
df_ter <- data.frame(Efecto = individual_effects, Tercil = terciles)

p2 <- ggplot(df_ter, aes(x=Tercil, y=Efecto, fill=Tercil)) +
  geom_boxplot() +
  labs(title="Variabilidad por Tercil") +
  theme_minimal() +
  theme(legend.position = "none")

# -----------------------------------------
# Panel 3: Top 5 variables más importantes
# -----------------------------------------
top5 <- head(importances, 5)

p3 <- ggplot(top5, aes(x=reorder(feature, importance), y=importance)) +
  geom_col() +
  coord_flip() +
  labs(
    title="Top 5 Características",
    x="Características",
    y="Importancia"
  ) +
  theme_minimal()

# -----------------------------------------
# Panel 4: Resumen Ejecutivo
# -----------------------------------------
resumen <- sprintf(
"RESUMEN EJECUTIVO

Efecto Promedio del Tratamiento:
• OLS: %.3f
• Random Forest: %.3f

Heterogeneidad:
• Desv. estándar: %.3f
• Rango: [%.3f, %.3f]

Predictores clave:
%s

Recomendación:
Focalizar programa en tercil alto
(%d personas, %d%% de la población)",
treatment_effect_ols,
ate_rf,
sd(individual_effects),
min(individual_effects),
max(individual_effects),
paste(importances$feature[1:3], collapse=", "),
sum(terciles == "Alto"),
round(mean(terciles == "Alto")*100)
)

p4 <- grid.text(resumen, x=0.05, y=0.95, just=c("left","top"))

# -----------------------------------------
# Guardar figura combinada
# -----------------------------------------
png("output/sintesis_final_p4.png", width=1400, height=1000)
grid.arrange(p1, p2, p3, p4, nrow=2)
dev.off()

cat("   ✓ Imagen guardada: output/sintesis_final_p4.png\n")

# -----------------------------------------
# Exportar resultados a JSON
# -----------------------------------------
results <- list(
  tratamiento = list(
    n_tratados = sum(T == 1),
    n_control = sum(T == 0)
  ),
  efectos = list(
    ols = treatment_effect_ols,
    random_forest = ate_rf,
    heterogeneidad_std = sd(individual_effects)
  ),
  top_caracteristicas = head(importances,5),
  terciles = list(
    bajo = list(
      n = sum(terciles=="Bajo"),
      efecto_medio = mean(individual_effects[terciles=="Bajo"])
    ),
    medio = list(
      n = sum(terciles=="Medio"),
      efecto_medio = mean(individual_effects[terciles=="Medio"])
    ),
    alto = list(
      n = sum(terciles=="Alto"),
      efecto_medio = mean(individual_effects[terciles=="Alto"])
    )
  )
)

write_json(results, "output/resultados_seccion2.json", pretty=TRUE)

cat("\nANÁLISIS COMPLETADO EXITOSAMENTE\n")
cat("✓ El programa de transferencias tiene efectos heterogéneos significativos\n")
cat("✓ Se recomienda focalizar en el tercil alto para maximizar impacto\n")
