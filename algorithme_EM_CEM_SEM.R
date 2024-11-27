################################################################################
#################### FITTING MIXTURES OF LINEAR REGRESSIONS ####################
################################################################################

rm(list = objects())
graphics.off()

#############################
### Jeu de données
####################

set.seed(1)

n = 200 # taille de notre jeu de données

intercept = rep(1, n)

# Variable 1 'V1' avec des valeurs tirées d'une distribution normale (N(10, 25))
V1 = runif(n, min=-1, max = 3)

X = as.matrix(data.frame(intercept=intercept, V1=V1))

# 2 régressions
Beta1 = as.matrix(c(0, 1))
Beta2 = as.matrix(c(4, 1))

Pi = c(0.3, 0.7)

Sigma1 = 1
Sigma2 = 1

e1 = rnorm(n*Pi[1], mean=0, sd=Sigma1)
e2 = rnorm(n*Pi[2], mean=0, sd=Sigma2)

# Y
Y = numeric(n)

Y[1:(n * Pi[1])] = X[1:(n * Pi[1]), ] %*% Beta1 + e1 # Pour les n1 premières observations
Y[(n * Pi[1] + 1):(n * Pi[1] + n * Pi[2])] = X[(n * Pi[1] + 1):(n * Pi[1] + n * Pi[2]), ] %*% Beta2 + e2 # Pour les n2 suivantes

df = data.frame(Y=Y, V1=V1)

# Aperçu du jeu de données et des droites de régression
# Calcul des prédictions pour les deux droites de régression
reg_1 = X %*% Beta1
reg_2 = X %*% Beta2

# Ajouter les prédictions comme colonnes au dataframe
df$reg_1 = reg_1
df$reg_2 = reg_2

# Utilisation de ggplot2 pour le nuage de points et les droites de régression
library(ggplot2)
ggplot(df, aes(x=V1, y=Y)) +
  geom_point(color="blue", alpha=0.6, size=2) + # Nuage de points
  geom_line(aes(y=reg_1, color="Régression 1"), linewidth=1) + # Droite de régression 1
  geom_line(aes(y=reg_2, color="Régression 2"), linewidth=1) + # Droite de régression 2
  scale_color_manual(values = c("Régression 1" = "red", "Régression 2" = "green")) + # Couleurs des droites
  labs(title="Nuage de points avec les deux droites de régression",
       x="V1",
       y="Y",
       color="Légende")



############################
### INITIALISATION
####################

beta = t(matrix(c(8, -1, 2, 1), ncol=2, byrow=TRUE))
sigma = runif(2, 0, 10)
pi1 = runif(1, 0.1, 0.9)
pi = c(pi1, 1 - pi1)



#########################
### EM
################

EM_algorithm <- function(fonction_densite, J, X, Y, nb_iterations,thresh=10^(-10)) {
  
  
  Q_previous = 0
  
  # Matrice des yi
  Y_Mat <- matrix(rep(Y, each = J), nrow = n, ncol = J, byrow = TRUE)
  
  for (iter in 1:nb_iterations) {
    # Calcul des densités pour chaque point et chaque composante
    densite <- dnorm(Y_Mat, mean = X%*%beta, 
                     sd = matrix(sigma, nrow = n, ncol = J, byrow = TRUE))
    densite_pondere <- t(t(densite) * pi)  # Densité pondérée
    
    # E-step : calcul des responsabilités
    Wij_mat <- densite_pondere  / rowSums(densite_pondere )
    
    
    # Step M
    for (j in 1:J) {
      W_j =  diag(Wij_mat[,j])
      
      # Mise à jour de beta[j]
      
      beta[, j] = solve(t(X) %*% W_j %*% X) %*% t(X) %*% W_j %*% Y
      
    }
    # Mise à jour de sigma
    sigma =sqrt(colSums(Wij_mat * (Y_Mat - X%*%beta)^2) / colSums(Wij_mat))
    
    # Mise à jour de pi
    pi = colSums(Wij_mat) / n
    pi = pi / sum(pi)  # Normalisation après mise a jour, pour que somme=1
    
    # Vérification de la convergence
    Q = sum(rowSums(Wij_mat * log(densite_pondere) ))
    Q_diff = abs(Q - Q_previous)  # Différence entre les vraisemblances
    
    if (Q_diff < thresh) {
      break
    }
    
    Q_previous = Q # Mettre à jour pour la prochaine itération
  }
  return (list(beta_hat=beta, sigma_hat=sigma, pi_hat=pi,nb_iter = iter, Q =Q))
}


# Exécution de l'algorithme CEM sur les données
results_EM = EM_algorithm(dnorm, J=2, X, Y, 10000)
results_EM





# Aperçu du jeu de données et des droites de régression true et pred
affiche_estim <- function(results,df, X, Y) {
  
    # Calcul des prédictions pour les deux droites de régression
    reg_1_hat <- X %*% results$beta_hat[,1]
    reg_2_hat <- X %*% results$beta_hat[,2]
    
    # Graphique 
    ggplot(df, aes(x=V1, y=Y)) +
      geom_point(color="blue", alpha=0.6, size=2) + # Nuage de points
      geom_line(aes(y=reg_1, color="Régression 1"), linewidth=1) + # Régression 1
      geom_line(aes(y=reg_2, color="Régression 2"), linewidth=1) + # Régression 2
      geom_line(aes(y=reg_1_hat, color="Régression prédite"), linewidth=1,linetype="dashed") +
      geom_line(aes(y=reg_2_hat, color="Régression prédite"), linewidth=1,linetype="dashed") +
      scale_color_manual(values=c("Régression 1"="red", "Régression 2"="green","Régression prédite"="black")) +
      labs(title="Nuage de points avec les deux droites de régression",
           x="V1",
           y="Y",
           color="Légende") 

}
affiche_estim(results_EM,df, X, Y)



############################
### INITIALISATION
####################

beta = t(matrix(c(10, 1, 2, 4), ncol=2, byrow=TRUE))
sigma = runif(2, 0, 10)
pi1 = runif(1, 0.1, 0.9)
pi = c(pi1, 1 - pi1)

#########################
### CEM
################

CEM_algorithm = function(fonction_densite, J, X, Y, nb_iterations,thresh= 10^(-10)){

  Q_previous = 0
  
  n = nrow(X)
  matrix_YY = matrix(rep(Y, each=J), nrow=n, ncol=J, byrow=TRUE)
  Z = matrix(0, nrow = n, ncol = 1)
  
  for (iter in 1:nb_iterations){
    # Step E
    # Calcul des densités pour chaque point et chaque composante
    densite = fonction_densite(matrix_YY, mean=X%*%beta, sd=matrix(sigma, nrow=n, ncol=J, 
                                                        byrow=TRUE))
    densite_pondere  = t(t(densite) * pi)  # Densité pondérée
    
    wij_matrix = densite_pondere  / rowSums(densite_pondere )
    
    
    # Step C
    for (i in 1:n) {
      w_i = wij_matrix[i, ] # proba des classes
      
      # Proba la plus forte pour chaque i
      argmax = max(w_i)
      indice_argmax = which(w_i == argmax)
      
      Z[i] = indice_argmax # classe
    }
    
    
    # Step M
    for (j in 1:J) {
      indices_classe_j = which(Z == j)
      
      # Mise à jour de pi[j]
      pi[j] = length(indices_classe_j) / n
      
      # Mise à jour de beta[j]
      X_j = X[indices_classe_j, ]
      Y_j = Y[indices_classe_j]
      w_nj = wij_matrix[indices_classe_j, j]
      W_j = diag(w_nj)
      beta[, j] = solve(t(X_j) %*% W_j %*% X_j) %*% t(X_j) %*% W_j %*% Y_j
      
      # Mise à jour de sigma[j]
      sigma_sq = sum(w_nj*(Y_j - X_j %*% beta[, j])^2) / sum(w_nj)
      sigma[j] = sqrt(sigma_sq)
    }
    pi = pi / sum(pi)  # Normalisation après mise a jour, pour que somme=1
    
    # Vérification de la convergence
    Q = sum(rowSums(wij_matrix * densite))
    Q_diff = abs(Q - Q_previous)  # Différence entre les vraisemblances
    
    if (Q_diff < thresh) {
      break
    }
    
    Q_previous = Q # Mettre à jour pour la prochaine itération
  }
  return (list(beta_hat=beta, sigma_hat=sigma, pi_hat=pi, nb_iter = iter,Q=Q))
}


# Exécution de l'algorithme CEM sur les données
results_CEM = CEM_algorithm(dnorm, J=2, X, Y, 10000)
results_CEM


affiche_estim(results_CEM,df, X, Y)





#########################
### SEM
################

# Prendre le nombre d'itération qu'il a fallu pour l'algo EM


SEM_algorithm = function(fonction_densite, J, X, Y, nb_iterations){
  
  n = nrow(X)
  matrix_YY = matrix(rep(Y, each=J), nrow=n, ncol=J, byrow=TRUE)
  Z = matrix(0, nrow = n, ncol = 1)
  
  for (iteration in 1:nb_iterations){
    # Step E
    # Calcul des densités pour chaque point et chaque composante
    densite = fonction_densite(matrix_YY, mean=X%*%beta, sd=matrix(sigma, nrow=n, ncol=J, 
                                                        byrow=TRUE))
    densite_pondere = t(t(densite) * pi)  # Densité pondérée
    
    wij_matrix = densite_pondere / rowSums(densite_pondere)
    
    
    # Step S
    for (i in 1:n) {
      w_i = wij_matrix[i, ] # proba des classes
      
      # Tirage multinomial pour chaque i
      tirage = rmultinom(1, size=1, prob=w_i)
      
      Z[i] = which(tirage == 1) # classe
    }
    
    
    # Step M
    for (j in 1:J) {
      indices_classe_j = which(Z == j)
      
      # Mise à jour de pi[j]
      pi[j] = length(indices_classe_j) / n
      
      # Mise à jour de beta[j]
      X_j = X[indices_classe_j, ]
      Y_j = Y[indices_classe_j]
      w_nj = wij_matrix[indices_classe_j, j]
      W_j = diag(w_nj)
      beta[, j] = solve(t(X_j) %*% W_j %*% X_j) %*% t(X_j) %*% W_j %*% Y_j
      
      # Mise à jour de sigma[j]
      sigma_sq = sum(w_nj*(Y_j - X_j %*% beta[, j])^2) / sum(w_nj)
      sigma[j] = sqrt(sigma_sq)
    }
    pi = pi / sum(pi)  # Normalisation après mise a jour, pour que somme=1
    Q = sum(rowSums(wij_matrix * densite))
  }
  return (list(beta_hat=beta, sigma_hat=sigma, pi_hat=pi,Q=Q))
}



# Exécution de l'algorithme SEM sur les données
results_SEM = SEM_algorithm(dnorm, J=2, X, Y, 100)
results_SEM


affiche_estim(results_SEM,df, X, Y)







