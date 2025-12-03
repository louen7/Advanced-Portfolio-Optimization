import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cp
import matplotlib.pyplot as plt
import scipy.stats as ss
from scipy.linalg import sqrtm
import riskfolio as rp
import scipy.optimize as sp
import time


n_samples = 1000  # Nombre de portefeuilles aléatoires pour la simulation Monte Carlo

# ------------------------------------------------------------------------------------------------
# 1. GESTION DES DONNÉES
# ------------------------------------------------------------------------------------------------

# Cette fonction permet de lire les données yfinance
# - 2016-2019 : "In-sample" pour les modeles d'entrainement
# - 2020-2021 : "Out-of-sample" pour les modèles dits "test"
def download_finance_data(n_assets=10):
    start = '2016-01-01'
    end = '2019-12-30'
    assets = ['JCI', 'TGT', 'CMCSA', 'CPB', 'MO', 'MMC', 'JPM',
              'ZION', 'PSA', 'BAX', 'BMY', 'LUV', 'PCAR', 'TXT', 'TMO',
              'MSFT', 'HPQ', 'SEE', 'VZ', 'CNP', 'NI', 'T', 'BA']
    assets.sort()
    if n_assets > 23:
        print('Warning: max number of assets is limited to 23')
        n_assets = 23

    training_data = yf.download(assets[:n_assets], start=start, end=end, group_by="ticker", auto_adjust=True)
    testing_data = yf.download(assets[:n_assets], start='2020-01-01', end='2021-12-30', group_by="ticker", auto_adjust=True)


    Y = dict()
    for ast in assets[:n_assets]:
        qq = training_data[ast]['Close']
        Y[ast] = [100*(qq.iloc[ii]- qq.iloc[ii-1])/qq.iloc[ii-1] for ii in range(1,len(qq))]
    training_df = pd.DataFrame(data=Y)

    Y = dict()
    for ast in assets[:n_assets]:
        qq = testing_data[ast]['Close']
        Y[ast] = [100 * (qq.iloc[ii] - qq.iloc[ii - 1]) / qq.iloc[ii - 1] for ii in range(1, len(qq))]
    testing_df = pd.DataFrame(data=Y)
    return training_df, testing_df


def compute_moments(y_data):

    mu = y_data.mean().to_numpy().reshape(1, -1)
    sigma = y_data.cov().to_numpy()
    return mu, sigma

def compute_metrics(x, training_df, testing_df):

    x = np.asarray(x)


    var0_scalar = x @ training_df.cov().to_numpy() @ x
    var0 = pd.DataFrame([var0_scalar])

    std0_scalar = np.sqrt(var0_scalar * 252)
    std0 = pd.DataFrame([std0_scalar])


    ret0_scalar = training_df.mean().to_numpy() @ x * 252
    ret0 = pd.DataFrame([ret0_scalar])


    stats_training = pd.concat([ret0, std0, var0], axis=0)
    stats_training.index = ['Return', 'Std. Dev.', 'Variance']


    var_scalar = x @ testing_df.cov().to_numpy() @ x
    var = pd.DataFrame([var_scalar])


    std_scalar = np.sqrt(var_scalar * 252)
    std = pd.DataFrame([std_scalar])


    ret = testing_df.mean().to_numpy() @ x * 252
    ret = pd.DataFrame([ret])


    stats_testing = pd.concat([ret, std, var], axis=0)
    stats_testing.index = ['Return', 'Std. Dev.', 'Variance']

    print('Training set: 2016 -- 2019')
    print(stats_training)
    print('Testing set: 2020-2021')
    print(stats_testing)

    return stats_training, stats_testing

# ------------------------------------------------------------------------------------------------
# 2. MODÈLES D'OPTIMISATION (MARKOWITZ)
# ------------------------------------------------------------------------------------------------

# Cette fonction résout le problème de portefeuille Markowitz standard (Question 5).
def markovitz_portfolio(mu, sigma, rmin=1.6/252):
    mu = np.asarray(mu).reshape(-1)
    n = mu.shape[0]
    x = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(x, sigma))
    constraints = [
        cp.sum(x) == 1, # Je m'assure que la somme des poids est 1.
        mu @ x >= rmin, # Je contrains le rendement à être supérieur ou égal à un minimum.
        x >= 0 # Je n'autorise pas la vente à découvert.
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return np.array(x.value).ravel()

# Ici, je mets en œuvre le portefeuille Markowitz probabiliste (Question 6).
# Je souhaite contrôler la probabilité pour que le rendement tombe sous un seuil alpha.
def markovitz_portfolio_probabilistic(mu, sigma, beta=0.49, alpha=1.6/252):
    # Je mets mu en vecteur (n,)
    mu = np.asarray(mu).reshape(-1)
    n = mu.shape[0]

    # Je pose la variable d'optimisation x
    x = cp.Variable(n)

    # Je verifie que la matrice est bien symetrique definie positive
    Sigma_reg = sigma + 1e-6 * np.eye(n)
    Sigma_sqrt = np.linalg.cholesky(Sigma_reg)


    # Je calcule le facteur k
    k = -ss.norm.ppf(beta)

    # Norme du risque (écart-type du portefeuille)
    risk_norm = cp.norm(Sigma_sqrt @ x, 2)

    # Je minimise la variance
    objective = cp.Minimize(cp.quad_form(x, sigma))

    # Contraintes :
    constraints = [
        cp.sum(x) == 1,          # Budget total (la somme des poids doit faire 1)
        x >= 0,                  # Pas de vente à découvert (les poids doivent être positifs)
        risk_norm <= (mu @ x - alpha) / k  # Je m'assure que le risque ne dépasse pas un certain seuil lié au rendement et à k.
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    if prob.status not in ["optimal", "optimal_inaccurate"] or x.value is None:
        print(f"Warning: Probabilistic Markowitz portfolio problem for {n} assets and beta={beta} is {prob.status}. Returning zero weights.")
        return np.zeros(n)
    else:
        return np.array(x.value).ravel()

# ------------------------------------------------------------------------------------------------
# 3. VISUALISATION
# ------------------------------------------------------------------------------------------------

# Je trace les séries temporelles des rendements pour visualiser la volatilité historique.
def plot_time_series_assets(Ytrain, Ytest):
    assets = Ytrain.columns[:3]  # seulement 3 actifs pour la lisibilité
    fig, axes = plt.subplots(2, 1, figsize=(14, 10)) # Augmenter la taille de la figure

    # TRAIN
    ax = axes[0]
    for a in assets:
        ax.plot(Ytrain[a], label=a, linewidth=2) # Lignes plus épaisses
    ax.set_title("Training data (2016–2019) - Returns (%)", fontsize=16, fontweight='bold')
    ax.set_ylabel("Return [%]", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=10, loc='best')
    ax.tick_params(axis='both', which='major', labelsize=10)

    # TEST
    ax = axes[1]
    for a in assets:
        ax.plot(Ytest[a], label=a, linewidth=2) # Lignes plus épaisses
    ax.set_title("Testing data (2020-2021) - Returns (%)", fontsize=16, fontweight='bold')
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Return [%]", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=10, loc='best')
    ax.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    plt.show()


def montecarlo_sim(num_assets, n_samples, Y):
    ####################################
    # Montecarlo Simulation
    ####################################

    rs = np.random.RandomState(seed=123)
    s1 = rs.dirichlet([0.1] * num_assets, n_samples)
    s2 = rs.dirichlet([0.25] * num_assets, n_samples)
    s3 = rs.dirichlet([0.5] * num_assets, n_samples)
    s4 = rs.dirichlet([0.75] * num_assets, n_samples)
    s5 = rs.dirichlet([1.0] * num_assets, n_samples)
    s6 = rs.dirichlet([1.5] * num_assets, n_samples)
    s7 = rs.dirichlet([2.0] * num_assets, n_samples)
    s8 = rs.dirichlet([3.0] * num_assets, n_samples)
    sample = np.concatenate([np.identity(num_assets), s1, s2, s3, s4, s5, s6, s7, s8], axis=0)

    m = sample.shape[0]
    M_1 = np.mean(Y.to_numpy(), axis=0).reshape(1, -1)
    M_2 = Y.cov().to_numpy()

    c_mean = 252 * M_1 @ sample.T
    c_var = np.zeros(m)


    for i in range(0, m):
        c_var[i] =  (252 * sample[i] @ M_2 @ sample[i].T) ** (0.5)


    return c_mean, c_var


def scatter_plot_port(c_mean, c_var, ret, std, title=''):
    ####################################
    # Plotting Portfolios
    ####################################

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax = np.ravel(ax)

    # Plotting Portfolios in mean-standard deviation plane
    cax0 = ax[0].scatter(c_var, c_mean, c=c_mean / c_var, cmap='Spectral', s=50, alpha=0.6)
    ax[0].scatter(std,
                  ret,
                  marker='*',
                  s=2 ** 8,
                  color='tab:red',
                  label='Computed solution')


    ax[0].set_xlim(min(c_var.min(), std) * 0.9, max(c_var.max(), std) * 1.1)
    ax[0].set_ylim(min(c_mean.min(), ret) * 0.9, max(c_mean.max(), ret) * 1.1)

    plt.xlabel('Standard deviation [%]', fontsize = 14)
    plt.ylabel('Return [%]', fontsize = 14)
    plt.grid()
    plt.legend()
    plt.title(title) 

    plt.show()

    return

# J'utilise cette fonction pour visualiser la répartition des poids (allocation) dans mon portefeuille.
def portfolio_composition_plot(Y, x):
    ####################################
    # Plotting Portfolios Composition
    ####################################


    df = dict()
    for i, elements in enumerate(Y):
        df[elements] = [x[i]] # Wrap scalar x[i] in a list

    dw = pd.DataFrame(data=df)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax = np.ravel(ax)

    rp.plot_pie(
        w=dw,
        title='Minimum Variance Portfolio',
        others=0.05,
        nrow=25,
        ax=ax[0]
    )

    fig.tight_layout()
    plt.show()

# ------------------------------------------------------------------------------------------------
# 4. OPTIMISATION MOMENTS D'ORDRE 4 (KURTOSIS)
# ------------------------------------------------------------------------------------------------


def compute_coefficients(Y, n):
    ####################################
    # Auxiliary functions
    ####################################



    def duplication_matrix(n):
        out = np.zeros((int(n * (n + 1) / 2), n ** 2))
        for j in range(1, n + 1):
            for i in range(j, n + 1):
                u = np.zeros((int(n * (n + 1) / 2), 1))
                u[round((j - 1) * n + i - ((j - 1) * j) / 2) - 1] = 1.0
                E = np.zeros((n, n))
                E[i - 1, j - 1] = 1.0
                E[j - 1, i - 1] = 1.0
                out += u @ E.reshape(-1, 1).T;
        return out.T


    def duplication_elimination_matrix(n):
        out = np.zeros((int(n * (n + 1) / 2), n ** 2))
        for j in range(n):
            e_j = np.zeros((1, n))
            e_j[0, j] = 1.0
            for i in range(j, n):
                u = np.zeros((int(n * (n + 1) / 2), 1))
                row = round(j * n + i - ((j + 1) * j) / 2)
                u[row] = 1.0
                e_i = np.zeros((1, n))
                e_i[0, i] = 1.0
                out += np.kron(u, np.kron(e_j, e_i))
        return out


    def kurt_matrix(Y):
        P = Y.to_numpy()
        T, n = P.shape
        mu = np.mean(P, axis=0).reshape(1, -1)
        mu = np.repeat(mu, T, axis=0)
        x = P - mu
        ones = np.ones((1, n))
        z = np.kron(ones, x) * np.kron(x, ones)
        S4 = 1 / T * z.T @ z
        return S4

    return duplication_matrix(n), duplication_elimination_matrix(n), kurt_matrix(Y)

# J'implémente un solveur SQP (Question 11).
def sqp_solve_q11(A, b, x0=None, max_iter=30, tol=1e-6, verbose=True):
    m, n = A.shape
    AT = A.T
    H = AT @ A  # Hessienne du problème


    if x0 is None:
        x0 = np.ones(n) * np.sqrt(0.5 / n)
        x0 = np.clip(x0, 0, 1)

    xk = x0.copy()

    grad_norm_hist = []
    constraint_hist = []
    step_norm_hist = []

    for k in range(max_iter):
        # Gradient de f(x) = 0.5 ||Ax-b||^2
        r = A @ xk - b
        grad_f = AT @ r

        # Contrainte g(x) = 0.5 - ||x||^2 <= 0
        g_xk = 0.5 - np.linalg.norm(xk) ** 2
        grad_g = -2 * xk

        # je pose la variable d'incrémentation
        dx = cp.Variable(n)

        # Je fixe l'objectif QP
        obj = cp.Minimize(grad_f @ dx + 0.5 * cp.quad_form(dx, H))

        # Contraintes :
        constraints = [
            xk + dx >= 0, # Je m'assure que les nouveaux poids restent positifs.
            xk + dx <= 1, # Je m'assure que les nouveaux poids ne dépassent pas 1.
            g_xk + grad_g @ dx <= 0
        ]

        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.SCS, verbose=False)

        if dx.value is None:
            if verbose:
                print(f"[Q11] Itération {k}: sous-problème QP infaisable ou échec du solveur.")
            break

        dx_val = dx.value
        xk1 = xk + dx_val

        # Mises à jour et historiques
        grad_norm = np.linalg.norm(grad_f)
        constraint_val = 0.5 - np.linalg.norm(xk1) ** 2  # doit être <= 0
        step_norm = np.linalg.norm(dx_val)

        grad_norm_hist.append(grad_norm)
        constraint_hist.append(constraint_val)
        step_norm_hist.append(step_norm)

        if verbose:
            print(f"[Q11] It {k}: ||grad f||={grad_norm:.3e}, "
                  f"g(x)={constraint_val:.3e}, ||dx||={step_norm:.3e}")

        # Je fixe le critère d'arrêt
        if step_norm < tol:
            if verbose:
                print(f"[Q11] Convergence atteinte à l'itération {k}.")
            xk = xk1
            break

        xk = xk1

    return xk, np.array(grad_norm_hist), np.array(constraint_hist), np.array(step_norm_hist)

# Je code un solveur SCP pour minimiser la Kurtosis (Question 13).
# J'utilise une région de confiance (delta) pour contrôler la taille du pas à chaque itération.
def kurtosis_scp_solve(L2, S2, S4, mu, x0_initial,
                       rmin=1.6/252,
                       max_iter=15,
                       tol=1e-4,
                       delta=0.1,  # Taille de la région de confiance
                       verbose=True):
    """
    Résout le problème de minimisation de la Kurtosis via Sequential Convex Programming (SCP).
    """

    # Préparation des données
    mu = np.asarray(mu).reshape(-1)
    n = mu.shape[0]
    p = int(n * (n + 1) / 2) # Dimension vecteur z

    # Calcul de la racine carrée de la matrice pour la contrainte SOC
    # || Sigma_4_sqrt * z || <= g
    Mat_inter = S2 @ S4 @ S2.T
    # Je m'assure la symétrie pour éviter les erreurs numériques
    Mat_inter = (Mat_inter + Mat_inter.T) / 2
    Sigma_4_sqrt = sqrtm(Mat_inter)

    if np.iscomplexobj(Sigma_4_sqrt):
        Sigma_4_sqrt = np.real(Sigma_4_sqrt)

    # J'initialise des variables au point k=0
    xk = np.asarray(x0_initial).reshape(-1)


    xk = np.maximum(xk, 0)
    xk = xk / np.sum(xk)

    Xk = np.outer(xk, xk)
    zk = L2 @ cp.vec(Xk).value # Utilisation de cp.vec pour être cohérent avec l'ordre
    gk = np.linalg.norm(Sigma_4_sqrt @ zk)

    hist_g = [gk]
    if verbose:
        print(f"Début SCP. g_initial = {gk:.6f}")

    # Boucle itérative SCP
    for it in range(max_iter):

        dx = cp.Variable(n)
        dX = cp.Variable((n, n), symmetric=True)
        dz = cp.Variable((p, 1))
        dg = cp.Variable()


        dx_col = cp.reshape(dx, (n, 1)) # Vecteur colonne
        dx_row = cp.reshape(dx, (1, n)) # Vecteur ligne

        # Variables au pas k+1 (Linéarisées)
        x_next = xk + dx
        X_next = Xk + dX
        z_next = zk.reshape(-1, 1) + dz
        g_next = gk + dg

        # Vecteur colonne pour les produits matriciels
        xk_col = xk.reshape(-1, 1)

        # --- Contraintes ---
        constraints = []


        constraints.append(
            X_next == xk_col @ xk_col.T + xk_col @ dx_row + dx_col @ xk_col.T # Je linéarise X_next
        )

        # Lien z et X
        # z = L2 * vec(X)
        constraints.append(
            z_next == L2 @ cp.reshape(cp.vec(X_next), (n*n, 1)) # Je lie z_next à la version linéarisée de X_next.
        )

        # Je fixe les contraintes de Kurtosis (SOC)
        # || Sigma_4_sqrt * z || <= g
        constraints.append(
            cp.SOC(g_next, Sigma_4_sqrt @ z_next)
        )

        # Je fixe les contraintes Portefeuille
        constraints.append(cp.sum(x_next) == 1) # La somme des poids doit être 1.
        constraints.append(x_next >= 0) # Pas de vente à découvert.
        constraints.append(mu @ x_next >= rmin) # Je maintiens la contrainte de rendement minimum.
        constraints.append(X_next >> 0) # Je m'assure que X_next est semi-définie positive.


        #  J'empêche le solveur de faire des sauts trop grands où la linéarisation pourrait etre fausse
        constraints.append(cp.norm(dx, 2) <= delta) # Je limite la taille du pas avec une région de confiance.


        # Je minimise g
        prob = cp.Problem(cp.Minimize(g_next), constraints)


        try:
            prob.solve(solver=cp.SCS, verbose=False, eps=1e-4)
        except Exception as e:
            if verbose:
                print(f"Erreur solveur à l'itération {it}: {e}")
            break

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            if verbose:
                print(f"Arrêt prématuré : Statut du solveur '{prob.status}' à l'it {it}")
            # Je réduis la region de confiance et on réessaie
            break

        # Récupération des valeurs
        dx_val = dx.value
        dg_val = dg.value
        step_size = np.linalg.norm(dx_val)

        # Mise à jour des points
        xk = xk + dx_val
        Xk = Xk + dX.value
        zk = zk + dz.value.ravel()
        gk = gk + dg_val

        hist_g.append(gk)

        if verbose:
            print(f"Iter {it+1:02d} | g = {gk:.6f} | ||dx|| = {step_size:.6f}")

        # Je fixe le critère d'arrêt
        if step_size < tol:
            if verbose:
                print("Convergence atteinte (taille du pas < tol).")
            break

    return xk, hist_g


def kurtosis_objective(x, L2, Sigma_4_sqrt):
    n = x.shape[0]
    X = np.outer(x, x)                     # X = xx^T
    z = L2 @ X.reshape(-1, 1)             # (p,1)
    v = Sigma_4_sqrt @ z                  # (p,1)
    return float(np.linalg.norm(v, 2) ** 2)


# J'utilise SLSQP de Scipy pour minimiser la Kurtosis (Question 14).
def kurtosis_slsqp(L2, S2, S4, mu, rmin=1.6/252, x0=None):

    mu = np.asarray(mu).reshape(-1)
    n = mu.shape[0]

    if x0 is None:
        x0 = np.ones(n) / n  # portefeuille pondéré de facon equitable

    Sigma_4_sqrt = sqrtm(S2 @ S4 @ S2.T)

    # Fonction objectif pour scipy
    def obj(x):
        return kurtosis_objective(x, L2, Sigma_4_sqrt)

    # Je fixe les contraintes
    cons = []

    # On sait que la somme des poids = 1
    cons.append(
        {
            'type': 'eq',
            'fun': lambda x: np.sum(x) - 1.0
        }
    )

    # mu^T x >= rmin  -> mu^T x - rmin >= 0
    cons.append(
        {
            'type': 'ineq',
            'fun': lambda x: mu @ x - rmin
        }
    )

    # Bornes 0 <= x_i <= 1
    bounds = [(0.0, 1.0) for _ in range(n)]

    # Appel à SLSQP
    res = sp.minimize(
        obj,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'maxiter': 500, 'ftol': 1e-9, 'disp': False}
    )

    return res.x, res

# J'utilise un probleme SDP pour résoudre le problème de minimisation de la Kurtosis (Question 17).
# Pour simplifier, au lieu de la contrainte difficile à prendre en compte X = xx^T, je pose X >= xx^T (relaxation convexe).
def kurtosis_sdp_relaxation(L2, S2, S4, mu, rmin=1.6/252):

    mu = np.asarray(mu).reshape(-1)
    n = mu.shape[0]
    p = int(n * (n + 1) / 2)

    Sigma_4_sqrt = sqrtm(S2 @ S4 @ S2.T)

    # Variables CVXPY
    x = cp.Variable(n)
    X = cp.Variable((n, n), symmetric=True)
    z = cp.Variable((p, 1))
    g = cp.Variable()

    # Matrice bloc pour la relaxation X ≈ xx^T via Schur
    # Je construis cette matrice bloc pour implémenter la relaxation SDP X >= xx^T.
    X_block = cp.bmat([
        [X, cp.reshape(x, (n, 1))],
        [cp.reshape(x, (1, n)), np.ones((1, 1))]
    ])

    constraints = []

    # Relaxation X ≈ xx^T
    constraints += [X_block >> 0] # Je souhaite que la matrice bloc soit semi-définie positive.

    # z = L2 vec(X)
    constraints += [
        z == L2 @ cp.reshape(cp.vec(X), (n * n, 1))
    ]

    # Je prend en compte la contrainte SOC : ||Sigma_4_sqrt z || <= g
    constraints += [
        cp.SOC(g, Sigma_4_sqrt @ z)
    ]

    #  J'implemente les contraintes de portefeuille
    constraints += [
        cp.sum(x) == 1,
        x >= 0,
        mu @ x >= rmin
    ]

    # On resout le problème d'optimisation
    objective = cp.Minimize(g)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    return np.array(x.value).ravel(), X.value, g.value, prob.status

print("All utility functions defined.")

####################################################################################################
# Bloc d'exécution (afficahe des graphiques et des résultats)
####################################################################################################

print("Début de l'exécution du script consolidé.")

# --- Q8 Execution ---
print("\n===== QUESTION 8 : Prix des actifs en fonction du temps =====")
n_assets_q8 = 3
Ytrain_q8, Ytest_q8 = download_finance_data(n_assets=n_assets_q8)
plot_time_series_assets(Ytrain_q8, Ytest_q8) # Je visualise les séries temporelles des actifs.

# --- Initial Q5 execution (utilisation de Ytrain_q8 et de Ytest_q8 for n_assets=3) ---
print("\n===== QUESTION 5 : Portefeuille Markowitz standard =====")
print("\n=== QUESTION 5 : n_assets = 3 ===")
# Je calcule les moments (mu, sigma)
mu_q5_initial, sigma_q5_initial = compute_moments(Ytrain_q8)
# J'optimise selon Markowitz
x_opt = markovitz_portfolio(mu_q5_initial, sigma_q5_initial)
print("Poids optimaux :", x_opt)
print("Somme des poids :", x_opt.sum())
print("Rendement espéré (train) :", (mu_q5_initial @ x_opt).item() * 252)
s_tr, s_te = compute_metrics(x_opt, Ytrain_q8, Ytest_q8) # Je calcule et affiche les métriques de performance.
print(f"Train : rendement moyen = {s_tr.loc['Return', 0]:.4f}%, variance = {s_tr.loc['Std. Dev.', 0]:.4f}")
print(f"Test  : rendement moyen = {s_te.loc['Return', 0]:.4f}%, variance = {s_te.loc['Std. Dev.', 0]:.4f}")


n_assets_3 = 3
Ytrain3 = Ytrain_q8
Ytest3 = Ytest_q8
mu3 = mu_q5_initial
sigma3 = sigma_q5_initial
x3 = x_opt
s_tr3 = s_tr
s_te3 = s_te
ret_tr3 = float(s_tr3.loc['Return'][0])
std_tr3 = float(s_tr3.loc['Std. Dev.'][0])
ret_te3 = float(s_te3.loc['Return'][0])
std_te3 = float(s_te3.loc['Std. Dev.'][0])

# --- Q5 execution ---
print("\n=== QUESTION 5 : variation du nombre d'actifs ===")
# Je teste l'impact de la variation en augmentant le nombre d'actifs
for k in [3, 5, 10, 15, 20]:
    print(f"\n--- n_assets = {k} ---")
    Ytrain_k, Ytest_k = download_finance_data(n_assets=k)
    mu_k, sigma_k = compute_moments(Ytrain_k)
    x_k = markovitz_portfolio(mu_k, sigma_k)
    s_tr_k, s_te_k = compute_metrics(x_k, Ytrain_k, Ytest_k)
    print(f"Train : rendement moyen = {s_tr_k.loc['Return', 0]:.4f}%, variance = {s_tr_k.loc['Std. Dev.', 0]:.4f}")
    print(f"Test  : rendement moyen = {s_te_k.loc['Return', 0]:.4f}%, variance = {s_te_k.loc['Std. Dev.', 0]:.4f}")

n_assets_20 = 20
Ytrain20 = Ytrain_k
Ytest20 = Ytest_k
mu20 = mu_k
sigma20 = sigma_k
x20 = x_k
s_tr20 = s_tr_k
s_te20 = s_te_k
ret_tr20 = float(s_tr20.loc['Return'][0])
std_tr20 = float(s_tr20.loc['Std. Dev.'][0])
ret_te20 = float(s_te20.loc['Return'][0])
std_te20 = float(s_te20.loc['Std. Dev.'][0])

# --- Q6 Partie 1: Probabilistic Markowitz pour 3 actifs ---
print("\n===== QUESTION 6 : Portefeuille Markowitz Probabiliste (Partie 1) =====")
print("\n=== Q6 : Portefeuille Markowitz Probabiliste pour 3 actifs ===")
# J'applique la méthode probabiliste (contrainte de probabilité de perte)
x_prob_marko = markovitz_portfolio_probabilistic(mu3, sigma3, beta=0.49, alpha=1.6/252)
print("\n--- Portefeuille Probabiliste Markowitz ---")
print("Poids optimaux (Probabiliste Markowitz) :", np.round(x_prob_marko, 4))
print("Somme des poids :", x_prob_marko.sum())
expected_return_prob_marko = mu3 @ x_prob_marko * 252
print("Rendement espéré (annuel) (Probabiliste Markowitz) : {:.4f}%".format(float(expected_return_prob_marko)))
s_tr_prob_marko, s_te_prob_marko = compute_metrics(x_prob_marko, Ytrain3, Ytest3)
print("\n--- Comparaison : Probabiliste Markowitz vs Standard Markowitz ---")
print("\nStandard Markowitz (3 actifs) :")
print("Poids optimaux       :", np.round(x3, 4))
print("Rendement espéré (annuel) (Train) : {:.4f}%".format(ret_tr3))
print("Ecart-type (annuel) (Train)      : {:.4f}%".format(std_tr3))
print("Rendement espéré (annuel) (Test)  : {:.4f}%".format(ret_te3))
print("Ecart-type (annuel) (Test)       : {:.4f}%".format(std_te3))
print("\nProbabiliste Markowitz (3 actifs) :")
print("Poids optimaux       :", np.round(x_prob_marko, 4))
print("Rendement espéré (annuel) (Train) : {:.4f}%".format(float(s_tr_prob_marko.loc['Return', 0])))
print("Ecart-type (annuel) (Train)      : {:.4f}%".format(float(s_tr_prob_marko.loc['Std. Dev.', 0])))
print("Rendement espéré (annuel) (Test)  : {:.4f}%".format(float(s_te_prob_marko.loc['Return', 0])))
print("Ecart-type (annuel) (Test)       : {:.4f}%".format(float(s_te_prob_marko.loc['Std. Dev.'][0])))


# --- Q6 Part 2: Analyse de la variation des actifs et de beta ---
print("\n===== QUESTION 6 : Analyse des actifs et de Beta (Partie 2) =====")
asset_counts = [5, 10, 20]
beta_values = [0.45, 0.4]
for n_assets_q6 in asset_counts:
    for beta_q6 in beta_values:
        print(f"\n--- Q6 : n_assets = {n_assets_q6}, beta = {beta_q6} ---")
        Ytrain_q6, Ytest_q6 = download_finance_data(n_assets=n_assets_q6)
        mu_q6, sigma_q6 = compute_moments(Ytrain_q6)
        x_prob_marko_q6 = markovitz_portfolio_probabilistic(mu_q6, sigma_q6, beta=beta_q6, alpha=1.6/252)
        print("Poids optimaux (Probabiliste Markowitz) :", np.round(x_prob_marko_q6, 4))
        print("Somme des poids :", x_prob_marko_q6.sum())
        expected_return_prob_marko_q6 = mu_q6 @ x_prob_marko_q6 * 252
        print("Rendement espéré (annuel) (Probabiliste Markowitz) : {:.4f}% ".format(float(expected_return_prob_marko_q6)))
        s_tr_prob_marko_q6, s_te_prob_marko_q6 = compute_metrics(x_prob_marko_q6, Ytrain_q6, Ytest_q6)

# --- Q7: Re évaluation des données avec 2021q---
print("\n===== QUESTION 7 : Réévaluation des portefeuilles avec données de test étendues =====")
# Download new data with extended test period (already handled by updated download_finance_data)
Ytrain_q7_3, Ytest_q7_3 = download_finance_data(n_assets=3)
Ytrain_q7_20, Ytest_q7_20 = download_finance_data(n_assets=20)

print("\n--- 3 Actifs (Q7) ---")
mu_q7_3, sigma_q7_3 = compute_moments(Ytrain_q7_3)
x_marko_q7_3 = markovitz_portfolio(mu_q7_3, sigma_q7_3)
print("\nStandard Markowitz (3 actifs):")
print("Poids optimaux:", np.round(x_marko_q7_3, 4))
print("Rendement espéré (annuel, train): {:.4f}%".format((mu_q7_3 @ x_marko_q7_3 * 252).item()))
s_tr_marko_q7_3, s_te_marko_q7_3 = compute_metrics(x_marko_q7_3, Ytrain_q7_3, Ytest_q7_3)
x_prob_marko_q7_3 = markovitz_portfolio_probabilistic(mu_q7_3, sigma_q7_3, beta=0.49, alpha=1.6/252)
print("\nProbabiliste Markowitz (3 actifs):")
print("Poids optimaux:", np.round(x_prob_marko_q7_3, 4))
print("Rendement espéré (annuel, train): {:.4f}%".format((mu_q7_3 @ x_prob_marko_q7_3 * 252).item()))
s_tr_prob_marko_q7_3, s_te_prob_marko_q7_3 = compute_metrics(x_prob_marko_q7_3, Ytrain_q7_3, Ytest_q7_3)

print("\n--- 20 Actifs (Q7) ---")
mu_q7_20, sigma_q7_20 = compute_moments(Ytrain_q7_20)
x_marko_q7_20 = markovitz_portfolio(mu_q7_20, sigma_q7_20)
print("\nStandard Markowitz (20 actifs):")
print("Poids optimaux:", np.round(x_marko_q7_20, 4))
print("Rendement espéré (annuel, train): {:.4f}%".format((mu_q7_20 @ x_marko_q7_20 * 252).item()))
s_tr_marko_q7_20, s_te_marko_q7_20 = compute_metrics(x_marko_q7_20, Ytrain_q7_20, Ytest_q7_20)
x_prob_marko_q7_20 = markovitz_portfolio_probabilistic(mu_q7_20, sigma_q7_20, beta=0.49, alpha=1.6/252)
print("\nProbabiliste Markowitz (20 actifs):")
print("Poids optimaux:", np.round(x_prob_marko_q7_20, 4))
print("Rendement espéré (annuel, train): {:.4f}%".format((mu_q7_20 @ x_prob_marko_q7_20 * 252).item()))
s_tr_prob_marko_q7_20, s_te_prob_marko_q7_20 = compute_metrics(x_prob_marko_q7_20, Ytrain_q7_20, Ytest_q7_20)
print("\n--- Discussion des observations (Q7) ---")
print("Avec l'extension des données de test jusqu'à fin 2021, on observe généralement que les rendements sur la période de test (2020-2021) sont plus faibles pour les deux types de portefeuilles comparés à la période d'entraánement (2016-2019).")
print("Ceci est particulièrement visible en 2020 et 2021, années marquées par une volatilité accrue et des événements économiques mondiaux (ex: COVID-19) qui ont impacté les marchés financiers. Les écarts-types sur la période de test sont également souvent plus élevés, reflétant cette volatilité.")
print("Il est intéressant de noter que, même si les rendements sont moindres, les poids des portefeuilles restent relativement stables, ce qui suggère que les stratégies d'optimisation basées sur les données d'entraánement génèrent des allocations robustes même face à des conditions de marché changeantes. Cependant, la performance réelle est impactée par la réalité du marché post-formation.")


# --- Q9 Monte Carlo Simulation ---
print("\n===== QUESTION 9 : Analyse Monte Carlo =====")
print("\n--- 1. Analyse avec 3 actifs ---")
print("Génération graph 3 actifs (Train)...")
plt.figure(figsize=(10, 6))
# Je lance la simulation Monte Carlo pour voir la frontière efficiente (Q9)
c_mean_tr3, c_var_tr3 = montecarlo_sim(n_assets_3, n_samples, Ytrain3)
scatter_plot_port(c_mean_tr3.ravel(), c_var_tr3, ret_tr3, std_tr3, title=f"Monte Carlo {n_assets_3} Actifs - Training (Frontière Efficiente)") 

print("Génération graph 3 actifs (Test)....")
plt.figure(figsize=(10, 6))
c_mean_te3, c_var_te3 = montecarlo_sim(n_assets_3, n_samples, Ytest3)
scatter_plot_port(c_mean_te3.ravel(), c_var_te3, ret_te3, std_te3, title=f"Monte Carlo {n_assets_3} Actifs - Testing (2020-2021)")


print("\n--- 2. Analyse avec 20 actifs ---")
print("Génération graph 20 actifs (Train)...")
c_mean_tr20, c_var_tr20 = montecarlo_sim(n_assets_20, n_samples, Ytrain20)
plt.figure(figsize=(10, 6))
scatter_plot_port(c_mean_tr20.ravel(), c_var_tr20, ret_tr20, std_tr20, title=f"Monte Carlo {n_assets_20} Actifs - Training (Diversification)")


print("Génération graph 20 actifs (Test)...")
c_mean_te20, c_var_te20 = montecarlo_sim(n_assets_20, n_samples, Ytest20)
plt.figure(figsize=(10, 6))
scatter_plot_port(c_mean_te20.ravel(), c_var_te20, ret_te20, std_te20, title=f"Monte Carlo {n_assets_20} Actifs - Testing (2020-2021)")


# --- Q11 Execution ---
print("\n===== QUESTION 11 : Résultats SQP (problème non-convexe) =====")
n_q11 = 20
m_q11 = 30
np.random.seed(1)
# Je génère un problème aléatoire pour tester mon algorithme SQP
A_q11 = np.random.randn(m_q11, n_q11)
b_q11 = np.random.randn(m_q11)
x_star, grad_hist, constr_hist, step_hist = sqp_solve_q11(A_q11, b_q11, verbose=True) # J'exécute le solveur SQP.
print("x* =", x_star)
print("||x*||^2 =", np.linalg.norm(x_star) ** 2)
print("Contrainte ||x||^2 >= 0.5 -> 0.5 - ||x||^2 =", 0.5 - np.linalg.norm(x_star) ** 2)
print("Contraintes 0 <= x <= 1 respectées ?",
      np.all(x_star >= -1e-6) and np.all(x_star <= 1 + 1e-6))
iters = np.arange(1, len(grad_hist) + 1)
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.semilogy(iters, grad_hist, marker='o')
plt.title("||∇f(x_k)|| ")
plt.xlabel("Itération")
plt.grid(True)
plt.subplot(1, 3, 2)
plt.plot(iters, constr_hist, marker='o')
plt.axhline(0, color='r', linestyle='--')
plt.title("g(x_k) = 0.5 - ||x||^2 (<= 0)")
plt.xlabel("Itération")
plt.grid(True)
plt.subplot(1, 3, 3)
plt.semilogy(iters, step_hist, marker='o')
plt.title("||dx|| ")
plt.xlabel("Itération")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Q14 Execution ---
print("\n===== QUESTION 14 : Résultats SLSQP (minimisation de la Kurtosis) =====")
n_assets_q14 = 3
Ytrain_q14, Ytest_q14 = download_finance_data(n_assets=n_assets_q14)
mu_q14, sigma_q14 = compute_moments(Ytrain_q14)
# Je calcule les matrices D2, L2, S4 nécessaires pour la Kurtosis
D2_q14, L2_q14, S4_q14 = compute_coefficients(Ytrain_q14, n_assets_q14)
S2_q14 = D2_q14.T @ D2_q14 @ L2_q14
x_marko_q14 = markovitz_portfolio(mu_q14, sigma_q14)
t0 = time.perf_counter()
x_slsqp, res_slsqp = kurtosis_slsqp(L2_q14, S2_q14, S4_q14, mu_q14, rmin=1.6/252, x0=x_marko_q14) # J'optimise la kurtosis avec SLSQP.
t1 = time.perf_counter()
time_slsqp = t1 - t0
print("Succès ?", res_slsqp.success)
print("Message :", res_slsqp.message)
print("Temps SLSQP : {:.4f} s".format(time_slsqp))
print("x_slsqp =", x_slsqp)
print("Somme des poids =", x_slsqp.sum())
print("Rendement (annuel) mu^T x_slsqp * 252 =",
      (mu_q14 @ x_slsqp).item() * 252)
print("Contraintes 0 <= x <= 1 respectées ?",
      np.all(x_slsqp >= -1e-6) and np.all(x_slsqp <= 1+1e-6))
print("Rendement >= rmin ?",
      (mu_q14 @ x_slsqp).item() >= 1.6/252 - 1e-6)
Sigma_4_sqrt_q14 = sqrtm(S2_q14 @ S4_q14 @ S2_q14.T)
kurt_slsqp = kurtosis_objective(x_slsqp, L2_q14, Sigma_4_sqrt_q14)
print("Valeur de l'objectif (kurtosis approx) SLSQP :", kurt_slsqp)
s_tr_slsqp, s_te_slsqp = compute_metrics(x_slsqp, Ytrain_q14, Ytest_q14) # Je compare les métriques.
s_tr_marko_q14, s_te_marko_q14 = compute_metrics(x_marko_q14, Ytrain_q14, Ytest_q14)
labels = ["Marko train", "SLSQP train", "Marko test", "SLSQP test"]
returns = [
    float(s_tr_marko_q14.loc["Return", 0]),
    float(s_tr_slsqp.loc["Return", 0]),
    float(s_te_marko_q14.loc["Return", 0]),
    float(s_te_slsqp.loc["Return", 0])
]
plt.figure(figsize=(7,4))
plt.bar(labels, returns)
plt.ylabel("Return [%]")
plt.title("Comparaison des rendements (Markowitz vs Kurtosis SLSQP)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# --- Q13 Execution ---
print("\n===== QUESTION 13 : SCP Kurtosis Optimisation =====")
x0_q13 = np.ones(n_assets_q14) / n_assets_q14
mu_q13_sim, sigma_q13_sim = compute_moments(Ytrain_q14)
# J'applique ma méthode SCP pour optimiser la Kurtosis et je trace la convergence
x_opt_scp, historique = kurtosis_scp_solve(L2_q14, S2_q14, S4_q14, mu_q13_sim, x0_q13, rmin=0.0, verbose=True)
plt.figure(figsize=(8, 5))
plt.plot(historique, 'o-', linewidth=2)
plt.title("Convergence de la fonction de coût (Kurtosis proxy)")
plt.xlabel("Itérations")
plt.ylabel("Valeur de g")
plt.grid(True)
plt.show()
print("\nPortfolio optimisé SCP :")
print(np.round(x_opt_scp, 4))
print("Somme poids :", np.sum(x_opt_scp))

# --- Q15 Monte Carlo + Pie charts (on utilise les données de la Q14) ---
print("\n===== QUESTION 15 : Monte Carlo + Pie charts =====")
s_tr_marko_q15, s_te_marko_q15 = compute_metrics(x_marko_q14, Ytrain_q14, Ytest_q14)
ret_marko_tr_q15 = float(s_tr_marko_q15.loc['Return'][0])
std_marko_tr_q15 = float(s_tr_marko_q15.loc['Std. Dev.'][0])

s_tr_kurt_q15, s_te_kurt_q15 = compute_metrics(x_slsqp, Ytrain_q14, Ytest_q14)
ret_kurt_tr_q15 = float(s_tr_kurt_q15.loc['Return'][0])
std_kurt_tr_q15 = float(s_tr_kurt_q15.loc['Std. Dev.'][0])

c_mean_tr_q15, c_var_tr_q15 = montecarlo_sim(n_assets_q14, n_samples, Ytrain_q14)
print("\n=== Q15 : Monte Carlo - Portefeuille Markowitz (TRAIN) ===")
scatter_plot_port(c_mean_tr_q15.ravel(), c_var_tr_q15, ret_marko_tr_q15, std_marko_tr_q15, title=f"Monte Carlo Markowitz {n_assets_q14} Actifs - Training") # Je visualise le portefeuille Markowitz sur la frontière efficiente.

c_mean_tr_k_q15, c_var_tr_k_q15 = montecarlo_sim(n_assets_q14, n_samples, Ytrain_q14)
print("\n=== Q15 : Monte Carlo - Portefeuille Kurtosis (TRAIN) ===")
scatter_plot_port(c_mean_tr_k_q15.ravel(), c_var_tr_k_q15, ret_kurt_tr_q15, std_kurt_tr_q15, title=f"Monte Carlo Kurtosis {n_assets_q14} Actifs - Training") # Je visualise le portefeuille Kurtosis sur la frontière efficiente.

asset_names_q15 = Ytrain_q14.columns
print("\n=== Q15 : Composition du portefeuille Markowitz ===")
portfolio_composition_plot(asset_names_q15, x_marko_q14) # Je montre la répartition des actifs pour Markowitz.
print("\n=== Q15 : Composition du portefeuille Kurtosis ===")
portfolio_composition_plot(asset_names_q15, x_slsqp) # Je montre la répartition des actifs pour la minimisation de Kurtosis.

# --- Q17 Execution ---
print("\n===== QUESTION 17 : Comparaison finale (Relaxation SDP) =====")
asset_list = [3, 5, 10, 20]
for n_assets_q17 in asset_list:
    print("\n==============================")
    print(f"=== QUESTION 17 : n_assets = {n_assets_q17} ===")
    print("==============================")

    Ytrain_q17, Ytest_q17 = download_finance_data(n_assets=n_assets_q17)
    mu_q17, sigma_q17 = compute_moments(Ytrain_q17)

    t0 = time.perf_counter()
    x_marko_q17 = markovitz_portfolio(mu_q17, sigma_q17)
    t1 = time.perf_counter()
    time_marko_q17 = t1 - t0

    print("\nPortefeuille Markowitz : ")
    print("x_marko =", x_marko_q17)
    print("Temps (Markowitz) : {:.4f} s".format(time_marko_q17))

    s_tr_marko_q17, s_te_marko_q17 = compute_metrics(x_marko_q17, Ytrain_q17, Ytest_q17)

    D2_q17, L2_q17, S4_q17 = compute_coefficients(Ytrain_q17, n_assets_q17)
    S2_q17 = D2_q17.T @ D2_q17 @ L2_q17

    t0 = time.perf_counter()
    # Je résous la relaxation SDP pour la Kurtosis
    x_sdp_q17, X_sdp_q17, g_sdp_q17, status_q17 = kurtosis_sdp_relaxation(L2_q17, S2_q17, S4_q17, mu_q17, rmin=1.6/252)
    t1 = time.perf_counter()
    time_sdp_q17 = t1 - t0

    print("\nPortefeuille SDP (relaxation convexe (10)) : ")
    print("Statut solveur :", status_q17)
    print("x_sdp =", x_sdp_q17)
    print("Temps (SDP relaxation) : {:.4f} s".format(time_sdp_q17))
    print("Somme des poids (SDP) =", x_sdp_q17.sum())
    print("µ^T x_sdp (annuel)   =", (mu_q17 @ x_sdp_q17).item() * 252)
    print("Valeur g* (kurtosis approx) =", g_sdp_q17)

    s_tr_sdp_q17, s_te_sdp_q17 = compute_metrics(x_sdp_q17, Ytrain_q17, Ytest_q17)

    X_xxt_q17 = np.outer(x_sdp_q17, x_sdp_q17)
    frob_diff_q17 = np.linalg.norm(X_sdp_q17 - X_xxt_q17, ord='fro')
    frob_X_q17 = np.linalg.norm(X_sdp_q17, ord='fro')

    print("\n--- Vérification de X = xx^T (relaxation) ---")
    print("||X - xx^T||_F =", frob_diff_q17)
    print("||X||_F        =", frob_X_q17)
    if frob_X_q17 > 1e-8:
        rel_q17 = frob_diff_q17 / frob_X_q17
        print("Rapport relatif ||X - xx^T||_F / ||X||_F = {:.4e}".format(rel_q17))
    else:
        rel_q17 = frob_diff_q17

    if rel_q17 < 1e-3:
        print("=> Ici, X ≈ xx^T : la relaxation semble quasi exacte (tight) pour cet exemple.")
    else:
        print("=> Ici, X n'est pas égal à xx^T : la relaxation SDP est plus lâche.")
        print("   On obtient seulement un borne inférieure sur la vraie kurtosis.")

    print("\n--- Comparaison temps / qualité ---")
    print("Temps Markowitz  : {:.4f} s".format(time_marko_q17))
    print("Temps SDP (Q17)  : {:.4f} s".format(time_sdp_q17))

    print("\nReturn / Std (train) :")
    print("Markowitz : Return = {:.4f}%, Std = {:.4f}".format(
        float(s_tr_marko_q17.loc['Return', 0]),
        float(s_tr_marko_q17.loc['Std. Dev.'][0])
    ))
    print("SDP (10)  : Return = {:.4f}%, Std = {:.4f}".format(
        float(s_tr_sdp_q17.loc['Return', 0]),
        float(s_tr_sdp_q17.loc['Std. Dev.'][0])
    ))

    print("\nReturn / Std (test) :")
    print("Markowitz : Return = {:.4f}%, Std = {:.4f}".format(
        float(s_te_marko_q17.loc['Return', 0]),
        float(s_te_marko_q17.loc['Std. Dev.'][0])
    ))
    print("SDP (10)  : Return = {:.4f}%, Std = {:.4f}".format(
        float(s_te_sdp_q17.loc['Return', 0]),
        float(s_te_sdp_q17.loc['Std. Dev.'][0])
    ))

print("\nFin du programme.")