clc; close all; clear all;
rng(0);

N = 4;
M = 3;
K = M;
sigma2 = 1;
gamma_target_linear = 10;
gamma_vec = gamma_target_linear * ones(K, 1);
P_dBm = 40;
P_watts = 10^((P_dBm - 30) / 10);
P_vec = P_watts * ones(N, 1);
H = (randn(N, K) + 1i * randn(N, K)) / 2;
M_scale = M;
link_noise_var = 0.1;

W_init = sqrt(P_watts/M) * (randn(N, M) + 1i * randn(N, M))/sqrt(2);
for i=1:N
    if norm(W_init(i,:))^2 > P_vec(i)
        W_init(i,:) = W_init(i,:) * sqrt(P_vec(i))/norm(W_init(i,:));
    end
end

max_ccp_iter = 15;
max_ppd_iter = 5000; 
Consensus_Loops = 1; 
Num_Duals = K + N;

fprintf('--- Running Damped Algorithm (eta decays) ---\n');

W_nodes = cell(N, 1); Mu_nodes = cell(N, 1);
for i = 1:N, W_nodes{i} = W_init; Mu_nodes{i} = zeros(Num_Duals, 1); end
hist_damped = [];

for ccp = 1:max_ccp_iter
    CCP_Weights = wheith(N);
    W_tensor_ccp = cat(3, W_nodes{:});
    W_flat_ccp = reshape(W_tensor_ccp, [], N);
    W_flat_ccp_new = W_flat_ccp * CCP_Weights.';
    W_prev_ccp = squeeze(num2cell(reshape(W_flat_ccp_new, N, M, N), [1 2]));
    
    for k = 1:max_ppd_iter
        step_size = 10 / ((1 + k) * ccp);
        eta_k = 1 / ((k + 1)^0.75);
        
        V_W = W_nodes; V_Mu = Mu_nodes;
        for c_round = 1:Consensus_Loops
            Current_Weights = wheith(N);
            
            W_tensor = cat(3, V_W{:}); W_flat = reshape(W_tensor, [], N);
            W_noise = (sqrt(link_noise_var/N)/sqrt(2)) * (randn(size(W_flat)) + 1i*randn(size(W_flat)));
            W_flat_avg = W_flat * Current_Weights.'; 
            W_flat_target = W_flat_avg + W_noise;
            W_flat_damped = (1 - eta_k) * W_flat + eta_k * W_flat_target;
            V_W = squeeze(num2cell(reshape(W_flat_damped, N, M, N), [1 2]));
            
            Mu_matrix = [V_Mu{:}];
            Mu_noise = sqrt(link_noise_var/N) * randn(size(Mu_matrix));
            Mu_matrix_avg = Mu_matrix * Current_Weights.';
            Mu_matrix_target = Mu_matrix_avg + Mu_noise;
            Mu_matrix_damped = (1 - eta_k) * Mu_matrix + eta_k * Mu_matrix_target;
            V_Mu = num2cell(Mu_matrix_damped, 1)';
        end
        
        [W_nodes, Mu_nodes] = run_subgradient(V_W, V_Mu, W_prev_ccp, N, M, K, P_vec, gamma_vec, H, M_scale, sigma2, step_size);
    end
    
    w_avg = zeros(N,M); for n=1:N, w_avg = w_avg + W_nodes{n}; end; w_avg = w_avg/N;
    curr_power = sum(sum(abs(w_avg).^2));
    hist_damped = [hist_damped; curr_power];
    fprintf('CCP %d: Damped Power = %.4f\n', ccp, curr_power);
end
w_final_damped = w_avg;

fprintf('\n--- Running Undamped Algorithm (eta = 1) ---\n');

W_nodes = cell(N, 1); Mu_nodes = cell(N, 1);
for i = 1:N, W_nodes{i} = W_init; Mu_nodes{i} = zeros(Num_Duals, 1); end
hist_undamped = [];

for ccp = 1:max_ccp_iter
    CCP_Weights = wheith(N);
    W_tensor_ccp = cat(3, W_nodes{:});
    W_flat_ccp = reshape(W_tensor_ccp, [], N);
    W_flat_ccp_new = W_flat_ccp * CCP_Weights.';
    W_prev_ccp = squeeze(num2cell(reshape(W_flat_ccp_new, N, M, N), [1 2]));

    for k = 1:max_ppd_iter
        step_size = 10 / ((1 + k) * ccp);
        eta_k = 1.0;
        
        V_W = W_nodes; V_Mu = Mu_nodes;
        for c_round = 1:Consensus_Loops
            Current_Weights = wheith(N);
            
            W_tensor = cat(3, V_W{:}); W_flat = reshape(W_tensor, [], N);
            W_noise = (sqrt(link_noise_var/N)/sqrt(2)) * (randn(size(W_flat)) + 1i*randn(size(W_flat)));
            W_flat_avg = W_flat * Current_Weights.'; 
            W_flat_target = W_flat_avg + W_noise;
            W_flat_damped = (1 - eta_k) * W_flat + eta_k * W_flat_target; 
            V_W = squeeze(num2cell(reshape(W_flat_damped, N, M, N), [1 2]));
            
            Mu_matrix = [V_Mu{:}];
            Mu_noise = sqrt(link_noise_var/N) * randn(size(Mu_matrix));
            Mu_matrix_avg = Mu_matrix * Current_Weights.';
            Mu_matrix_target = Mu_matrix_avg + Mu_noise;
            Mu_matrix_damped = (1 - eta_k) * Mu_matrix + eta_k * Mu_matrix_target; 
            V_Mu = num2cell(Mu_matrix_damped, 1)';
        end
        
        [W_nodes, Mu_nodes] = run_subgradient(V_W, V_Mu, W_prev_ccp, N, M, K, P_vec, gamma_vec, H, M_scale, sigma2, step_size);
    end
    
    w_avg = zeros(N,M); for n=1:N, w_avg = w_avg + W_nodes{n}; end; w_avg = w_avg/N;
    curr_power = sum(sum(abs(w_avg).^2));
    hist_undamped = [hist_undamped; curr_power];
    fprintf('CCP %d: Undamped Power = %.4f\n', ccp, curr_power);
end
w_final_undamped = w_avg;

fprintf('\n--- Running Centralized CVX Benchmark ---\n');

W_cvx_prev = W_init; 
hist_cvx = [];

for ccp = 1:max_ccp_iter
    cvx_begin quiet
        variable W_cvx(N, M) complex
        minimize( square_pos(norm(W_cvx, 'fro')) )
        subject to
            for n = 1:N, sum_square_abs(W_cvx(n,:)) <= P_vec(n); end
            for k_user = 1:K
                interf_cvx = sigma2;
                for j = 1:M, if j ~= k_user, interf_cvx = interf_cvx + square_abs(H(:,k_user)' * W_cvx(:,j)); end; end
                
                h_wk_prev = H(:,k_user)' * W_cvx_prev(:,k_user);
                h_wk_curr = H(:,k_user)' * W_cvx(:,k_user);
                sig_lin_cvx = 2 * real(conj(h_wk_prev) * h_wk_curr) - abs(h_wk_prev)^2;
                
                gamma_vec(k_user) * interf_cvx <= sig_lin_cvx;
            end
    cvx_end
    
    if strcmpi(cvx_status, 'Solved')
        curr_pow = sum(sum(abs(W_cvx).^2));
        hist_cvx = [hist_cvx; curr_pow];
        W_cvx_prev = W_cvx;
    else
        hist_cvx = [hist_cvx; NaN];
    end
    fprintf('CCP %d: CVX Power = %.4f\n', ccp, hist_cvx(end));
end

figure;
semilogy(hist_damped, '-s', 'LineWidth', 2, 'DisplayName', 'Damped (\eta_k decay)'); hold on;
semilogy(hist_undamped, '--x', 'LineWidth', 2, 'DisplayName', 'Undamped (\eta_k = 1)');
semilogy(hist_cvx, '-o', 'LineWidth', 2, 'DisplayName', 'CVX Benchmark');
grid on; legend;
xlabel('CCP Iteration'); ylabel('Total Power (Watts)');
title('Effect of Noise Damping on Convergence');

verify_constraints('DAMPED (Robust)', w_final_damped, N, M, K, P_vec, gamma_vec, H, sigma2);
verify_constraints('UNDAMPED (Noisy)', w_final_undamped, N, M, K, P_vec, gamma_vec, H, sigma2);

function [W_next, Mu_next] = run_subgradient(V_W, V_Mu, W_prev_ccp, N, M, K, P_vec, gamma_vec, H, M_scale, sigma2, step_size)
    W_next = cell(N,1); Mu_next = cell(N,1);
    Num_Duals = K + N;
    
    for i = 1:N
        vx = V_W{i}; vmu = V_Mu{i}; w_t = W_prev_ccp{i};
        
        Grad_Obj = zeros(N, M); Grad_Obj(i, :) = 2 * vx(i, :);
        Grad_Penalty = zeros(N, M);
        
        for u = 1:K
            interf = sigma2;
            for j = 1:M, if j~=u, interf = interf + abs(H(:,u)'*vx(:,j))^2; end; end
            h_wt = H(:,u)'*w_t(:,u); h_w  = H(:,u)'*vx(:,u);
            sig_lin = 2*real(conj(h_wt)*h_w) - abs(h_wt)^2;
            
            if (gamma_vec(u)*interf - sig_lin) > 0
                mu_val = vmu(u);
                for j=1:M
                    if j~=u, h_u=H(:,u); Grad_Penalty(:,j) = Grad_Penalty(:,j) + mu_val*2*gamma_vec(u)*h_u*(h_u'*vx(:,j)); end
                end
                h_u=H(:,u); Grad_Penalty(:,u) = Grad_Penalty(:,u) + mu_val*(-2)*h_u*(h_u'*w_t(:,u));
            end
        end
        
        for n_idx = 1:N
            if (norm(vx(n_idx,:))^2 - P_vec(n_idx)) > 0
                mu_idx = K + n_idx;
                Grad_Penalty(n_idx,:) = Grad_Penalty(n_idx,:) + vmu(mu_idx)*2*vx(n_idx,:);
            end
        end
        
        D_total = Grad_Obj + M_scale * Grad_Penalty;
        grad_norm = norm(D_total, 2);
        if grad_norm > 100, D_total = D_total * (100/grad_norm); end
        
        W_temp = vx - step_size * D_total;
        if norm(W_temp(i,:)) > sqrt(P_vec(i)), W_temp(i,:) = W_temp(i,:) * (sqrt(P_vec(i))/norm(W_temp(i,:))); end
        W_next{i} = W_temp;
        
        G_vec = zeros(Num_Duals, 1);
        for u=1:K
            interf = sigma2;
            for j=1:M, if j~=u, interf = interf + abs(H(:,u)'*vx(:,j))^2; end; end
            h_wt = H(:,u)'*w_t(:,u); h_w = H(:,u)'*vx(:,u);
            sig_lin = 2*real(conj(h_wt)*h_w) - abs(h_wt)^2;
            G_vec(u) = gamma_vec(u)*interf - sig_lin;
        end
        for n_idx=1:N, G_vec(K+n_idx) = norm(vx(n_idx,:))^2 - P_vec(n_idx); end
        Mu_next{i} = max(0, vmu + step_size * G_vec);
    end
end

function verify_constraints(label, w_final, N, M, K, P_vec, gamma_vec, H, sigma2)
    fprintf('\n==============================================\n');
    fprintf('   VERIFICATION: %s\n', label);
    fprintf('==============================================\n');
    
    per_antenna_power = sum(abs(w_final).^2, 2);
    sat_power = all(per_antenna_power <= P_vec + 1e-4);
    if sat_power, fprintf('>> Power Constraints: SATISFIED\n'); else, fprintf('>> Power Constraints: VIOLATED\n'); end
    
    sat_sinr = true;
    for k=1:K
        sig = abs(H(:,k)' * w_final(:,k))^2;
        interf = 0;
        for j=1:M, if j ~= k, interf = interf + abs(H(:,k)' * w_final(:,j))^2; end; end
        val = sig / (sigma2 + interf);
        if val < gamma_vec(k) - 1e-4, sat_sinr = false; end
    end
    if sat_sinr, fprintf('>> SINR Constraints:  SATISFIED\n'); else, fprintf('>> SINR Constraints:  VIOLATED\n'); end
end

function W = wheith(N)
     W = (1/N) * ones(N,N);
end
