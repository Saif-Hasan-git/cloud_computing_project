import numpy as np
import random
import pandas as pd
import requests
from io import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Q-learning parameters
ALPHA, GAMMA, EPSILON = 0.1, 0.9, 0.2
EPISODES = 3000

# Reward weights
COST_W, TIME_W = 0.6, 0.4

def initialize_env(num_vms, num_apps):
    vm_cost = np.random.uniform(0.6, 1.0, num_vms)
    vm_mips = np.random.randint(1500, 3000, num_vms)
    app_mi = np.random.randint(5000, 15000, num_apps)
    return vm_cost, vm_mips, app_mi

def reward_calc(app_mi, vm_mips, vm_cost, cur_time, app_i, vm_i):
    t = app_mi[app_i]/vm_mips[vm_i]
    cost = t*vm_cost[vm_i]
    finish = cur_time[vm_i]+t
    score = COST_W*cost + TIME_W*finish
    return -score, t

def train_qtable(nv, na):
    Q = np.zeros((na, nv))
    for _ in range(EPISODES):
        vm_cost, vm_mips, app_mi = initialize_env(nv, na)
        cur_time = np.zeros(nv)
        for ai in range(na):
            if random.random() < EPSILON:
                vi = random.randrange(nv)
            else:
                vi = np.argmin(Q[ai])
            r, t = reward_calc(app_mi, vm_mips, vm_cost, cur_time, ai, vi)
            cur_time[vi] += t
            Q[ai, vi] += ALPHA * (r - Q[ai, vi])
    return Q

def generate_synthetic(Q, ntests, nv, na):
    rows = []
    for tid in range(ntests):
        vm_cost, vm_mips, app_mi = initialize_env(nv, na)
        cur_time = np.zeros(nv)
        for ai in range(na):
            vi = np.argmin(Q[ai])
            t = app_mi[ai]/vm_mips[vi]
            cur_time[vi] += t
            rows.append({
                **{f'VM_MIPS_{j}': vm_mips[j] for j in range(nv)},
                **{f'VM_Cost_{j}': vm_cost[j] for j in range(nv)},
                'App_Size': app_mi[ai],
                'Assigned_VM': vi,
                'Cur_Time': cur_time[vi],
                'Test_ID': tid
            })
    return pd.DataFrame(rows)

def build_ml_classifier(df, num_vms):
    features = ['App_Size'] + [f'VM_MIPS_{j}' for j in range(num_vms)] + [f'VM_Cost_{j}' for j in range(num_vms)]
    X = df[features]
    y = df['Assigned_VM']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\nML Classification Report:\n", classification_report(y_test, y_pred))
    print(f"Accuracy: {acc * 100:.2f}%")

    joblib.dump((clf, features), "vm_load_balancer_model.pkl")
    return clf, features

def predict_vm(model, app_size, vm_mips, vm_cost, features):
    num_vms = len(vm_mips)
    X_input = pd.DataFrame([{**{f'VM_MIPS_{j}': vm_mips[j] for j in range(num_vms)},
                             **{f'VM_Cost_{j}': vm_cost[j] for j in range(num_vms)},
                             'App_Size': app_size}])[features]  # enforce column order
    return model.predict(X_input)[0]

def plot_metrics(df, nv):
    df_time = df.groupby('Test_ID')['Cur_Time'].max()
    df_cost = df.groupby('Test_ID').apply(lambda d: sum(d.Cur_Time * [d[f'VM_Cost_{v}'].iloc[0] for v in d.Assigned_VM]))
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(df_time, label='Total Time')
    plt.xlabel('Test ID'); plt.ylabel('Time'); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(df_cost, label='Total Cost', color='orange')
    plt.xlabel('Test ID'); plt.ylabel('Cost'); plt.legend()
    plt.tight_layout()
    plt.show()

def plot_vm_utilization(df, num_vms):
    vm_util = {f'VM_{i}': 0 for i in range(num_vms)}
    total_time = df.groupby('Test_ID')['Cur_Time'].max().sum()

    for i in range(num_vms):
        vm_i_df = df[df['Assigned_VM'] == i]
        vm_util[f'VM_{i}'] = vm_i_df['Cur_Time'].sum() / total_time * 100

    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(vm_util.keys()), y=list(vm_util.values()), palette='Set2')
    plt.ylabel('Utilization (%)')
    plt.title('VM Utilization Across All Tests')
    plt.show()

def plot_app_distribution(df, num_vms):
    vm_indices = list(range(num_vms))
    counts_series = df['Assigned_VM'].value_counts().sort_index()
    full_counts = [counts_series.get(i, 0) for i in vm_indices]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=[f'VM_{i}' for i in vm_indices], y=full_counts, palette='coolwarm')
    plt.ylabel('Number of Applications Assigned')
    plt.title('Application Distribution Across VMs')
    plt.show()

def plot_execution_time_box(df):
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Assigned_VM', y='Cur_Time', data=df, palette='pastel')
    plt.ylabel('Execution Time (s)')
    plt.xlabel('Assigned VM')
    plt.title('Execution Time Distribution per VM')
    plt.show()

def plot_assignment_heatmap(df, num_vms):
    heatmap_df = pd.crosstab(df['Test_ID'], df['Assigned_VM'])
    heatmap_df = heatmap_df.reindex(columns=range(num_vms), fill_value=0)
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_df, cmap='YlGnBu', cbar=True, linewidths=0.5)
    plt.xlabel("VM Index")
    plt.ylabel("Test ID")
    plt.title("Heatmap of Application Assignment per Test")
    plt.show()

def compute_resource_sharing_index(df, num_vms):
    counts = df['Assigned_VM'].value_counts()
    avg = counts.mean()
    imbalance = sum(abs(c - avg) for c in counts)
    rsi = 1 - (imbalance / (len(df)))
    print(f"Resource Sharing Index (0–1): {rsi:.3f}")
    return rsi

def analyze(df, num_vms=3):
    compute_resource_sharing_index(df, num_vms)
    plot_vm_utilization(df, num_vms)
    plot_app_distribution(df, num_vms)
    plot_execution_time_box(df)
    plot_assignment_heatmap(df, num_vms)

def main():
    num_vms, num_apps, num_tests = 3, 5, 300
    print("Training Q-learning agent...")
    Q = train_qtable(num_vms, num_apps)

    print("Generating synthetic dataset...")
    df = generate_synthetic(Q, num_tests, num_vms, num_apps)

    print("Training ML classifier...")
    model, features = build_ml_classifier(df, num_vms)

    df.to_csv("combined_lb_dataset.csv", index=False)
    print("Saved dataset to 'combined_lb_dataset.csv'")

    print("\n--- Live Deployment Test ---")
    vm_cost = np.random.uniform(0.1, 1.0, num_vms)
    vm_mips = np.random.randint(1000, 3000, num_vms)
    for i in range(num_apps):
        app_size = np.random.randint(1000, 10000)
        assigned = predict_vm(model, app_size, vm_mips, vm_cost, features)
        print(f"App {i} (Size: {app_size}) assigned to VM {assigned}")

    print("\nPlotting synthetic metrics...")
    plot_metrics(df, num_vms)
    analyze(df, num_vms)

if __name__ == "__main__":
    main()