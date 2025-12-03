import numpy as np
import cvxpy as cp
import time
import matplotlib.pyplot as plt
from scipy.optimize import minimize
#J'ai mis les questions 1, 2 ET 3 dans ce fichier car le code met beaucoup de temps à tourner 
# QUESTION 1

# Je définis les dimensions de mon problème
n = 20   # taille du vecteur x
m = 30   # nombre de lignes de A

# Je fixe une "seed" pour la reproductibilité des résultats
np.random.seed(1)

# Je génère la matrice A de taille m x n et le vecteur b de taille m aléatoirement
A = np.random.randn(m, n)
b = np.random.randn(m)

# Je définis ma variable de décision x (un vecteur de taille n) avec CVXPY
x = cp.Variable(n)

# Je définis la fonction que je veux minimiser : 
objective = cp.Minimize(0.5 * cp.sum_squares(A @ x - b))

# Je spécifie les contraintes : chaque composante de x doit être entre 0 et 1 (bornes)
constraints = [x >= 0, x <= 1]

# Je construis le problème d'optimisation avec mon objectif et mes contraintes
prob = cp.Problem(objective, constraints)

# Je résous le problème
prob.solve()

# J'affiche les résultats pour la Question 1
print("=== QUESTION 1 ===")
print("Status du problème :", prob.status)
print("Valeur optimale :", prob.value)

# Je récupère la solution optimale pour x
x_opt = x.value

X = x_opt.reshape(4, 5)

# Je configure l'affichage pour avoir 6 décimales
np.set_printoptions(precision=6, suppress=False)

print("\nSolution x* au format matrice 4x5 :")
print(X)



# QUESTION 2 : étude du temps de calcul
print("\n\n=== QUESTION 2 : étude du temps de calcul ===")

# Je choisis différentes tailles pour n, allant de 20 à 2000
ns = np.linspace(20, 2000, 8, dtype=int)

# Je définis m comme étant proportionnel à n (environ 1.5 fois n)
ms = (1.5 * ns).astype(int)

# Je crée des listes pour stocker les temps de résolution et les statuts des problèmes
solve_times = []
statuses = []

# Je boucle sur chaque paire (n, m) pour résoudre le problème et mesurer le temps
for n_i, m_i in zip(ns, ms):
    print(f"\nRésolution pour n = {n_i}, m = {m_i}...")

    # Je génère de nouvelles données (A et b) pour chaque taille de problème
    A = np.random.randn(m_i, n_i)
    b = np.random.randn(m_i)

    # Je crée une nouvelle variable x avec la taille n_i actuelle
    x = cp.Variable(n_i)

    objective = cp.Minimize(0.5 * cp.sum_squares(A @ x - b))

    # Je définis les contraintes (toujours 0 <= x <= 1)
    constraints = [x >= 0, x <= 1]

    # Je construis le nouveau problème
    prob = cp.Problem(objective, constraints)

    # J'essaye plusieurs solveurs au cas où certains échouent
    solvers_to_try = [None, cp.ECOS, cp.OSQP, cp.SCS]
    solved = False
    dt = None  

    for s in solvers_to_try:
        try:
            if s is None:
                # Si le solveur est None, je laisse CVXPY choisir par défaut
                print("  -> essai avec le solveur par défaut...")
                t0 = time.perf_counter()
                prob.solve()
                t1 = time.perf_counter()
            else:
                # Sinon, j'impose un solveur spécifique
                print(f"  -> essai avec le solveur {s} ...")
                t0 = time.perf_counter()
                prob.solve(solver=s)
                t1 = time.perf_counter()

            # Je vérifie si le problème a été résolu de manière satisfaisante
            if prob.status not in ["infeasible", "unbounded", "infeasible_inaccurate"]:
                dt = t1 - t0
                print(f"     OK, status = {prob.status}, temps = {dt:.3f} s")
                solved = True
                break  # Si ça a marché, je passe au problème suivant
            else:
                print(f"     Problème avec ce solveur, status = {prob.status}")
        except Exception as e:
            print(f"     Erreur avec ce solveur : {e}")

    # Si aucun solveur n'a réussi pour cette taille de problème
    if not solved:
        print("  -> Aucun solveur n'a réussi pour cette taille.")
        solve_times.append(np.nan) # J'ajoute NaN car la résolution a échoué
        statuses.append("failed")
    else:
        # J'enregistre le temps de résolution et le statut
        solve_times.append(dt)
        statuses.append(prob.status)



# TRACÉS 


valid_ns = [n for n, st in zip(ns, statuses) if st != "failed"]
valid_ms = [m for m, st in zip(ms, statuses) if st != "failed"]
valid_times = [t for t, st in zip(solve_times, statuses) if st != "failed"]

# Si j'ai au moins un point valide, je trace les courbes
if len(valid_times) > 0:
    plt.figure()
    plt.plot(valid_ns, valid_times, marker="o")
    plt.xlabel("n (dimension de x)")
    plt.ylabel("Temps de résolution (s)")
    plt.title("Temps de résolution en fonction de n (m \u2248 1.5 n)")
    plt.grid(True)

    plt.figure()
    plt.plot(valid_ms, valid_times, marker="o")
    plt.xlabel("m (nombre de lignes de A)")
    plt.ylabel("Temps de résolution (s)")
    plt.title("Temps de résolution en fonction de m (m \u2248 1.5 n)")
    plt.grid(True)

    plt.show()
else:
    # Si rien n'a marché, j'affiche un message d'information, ce qui n'est peu probable
    print("\nAucune taille n'a pu être résolue correctement.")


# QUESTION 3 :


print("\n\n=== QUESTION 3 : comparaison CVXPY vs SciPy (SLSQP) ===")

times_cvxpy = []
times_slsqp = []


diff_solutions = []

for n_i, m_i in zip(ns, ms):
    print(f"\nTaille du problème : n = {n_i}, m = {m_i}")

    # Je génère de nouvelles données A et b
    A = np.random.randn(m_i, n_i)
    b = np.random.randn(m_i)

    # Je définis la variable CVXPY
    x_cvx = cp.Variable(n_i)

    # Je construis l'objectif pour CVXPY
    objective = cp.Minimize(0.5 * cp.sum_squares(A @ x_cvx - b))

    # Et les contraintes
    constraints = [x_cvx >= 0, x_cvx <= 1]

    # Je définis le problème CVXPY
    prob = cp.Problem(objective, constraints)

    t0 = time.perf_counter()
    prob.solve()
    t1 = time.perf_counter()

    dt_cvx = t1 - t0
    times_cvxpy.append(dt_cvx)

    x_cvx_val = x_cvx.value  # Je récupère la solution de CVXPY
    print(f"  CVXPY : status = {prob.status}, temps = {dt_cvx:.3f} s")

    def fun(x):
        r = A @ x - b
        return 0.5 * np.dot(r, r)

    def grad(x):
        r = A @ x - b
        return A.T @ r

    x0 = np.zeros(n_i)

    # Je définis les bornes pour les variables pour SciPy (entre 0 et 1 pour chaque composante)
    bounds = [(0.0, 1.0) for _ in range(n_i)]

    # Je lance la minimisation avec la méthode SLSQP de SciPy.optimize et je mesure le temps
    t2 = time.perf_counter()
    res = minimize(
        fun,
        x0,
        method='SLSQP',
        jac=grad,             # Je lui donne le gradient pour une meilleure performance
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-9} # cela peut etre ajusté si je le souhaite
    )
    t3 = time.perf_counter()

    dt_slsqp = t3 - t2
    times_slsqp.append(dt_slsqp)

    x_slsqp_val = res.x   # Je récupère la solution trouvée par SLSQP

    print(f"  SciPy SLSQP : success = {res.success}, temps = {dt_slsqp:.3f} s")

    # Je compare les solutions de CVXPY et SciPy
    if x_cvx_val is not None:
        diff = np.linalg.norm(x_cvx_val - x_slsqp_val)
        diff_solutions.append(diff)
        print(f"  ||x_cvxpy - x_slsqp||_2 = {diff:.3e}")
    else:
        diff_solutions.append(np.nan)
        print("  Impossible de comparer les solutions (CVXPY n'a pas renvoyé de solution).")

times_cvxpy = np.array(times_cvxpy)
times_slsqp = np.array(times_slsqp)

# Je trace la comparaison des temps de résolution en fonction de n
plt.figure()
plt.plot(ns, times_cvxpy, marker="o", label="CVXPY")
plt.plot(ns, times_slsqp, marker="s", label="SciPy SLSQP")
plt.xlabel("n (dimension de x)")
plt.ylabel("Temps de résolution (s)")
plt.title("Comparaison des temps de calcul : CVXPY vs SciPy SLSQP")
plt.grid(True)
plt.legend()
plt.show()


print("\nRésumé des différences de solutions (norme L2) :")
for n_i, m_i, d in zip(ns, ms, diff_solutions):
    print(f"n = {n_i:4d}, m = {m_i:4d} -> ||x_cvxpy - x_slsqp||_2 = {d:.3e}")
