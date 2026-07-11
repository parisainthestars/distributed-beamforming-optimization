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

W_init = sqrt(P_watts/M) * (randn(N, M) + 1i * randn(N, M))/sqrt(2);
for i=1:N
    if norm(W_init(i,:))^2 > P_vec(i)
        W_init(i,:) = W_init(i,:) * sqrt(P_vec(i))/norm(W_init(i,:));
    end
end

max_ccp_iter = 15;
max_ppd_iter = 15000;
Consensus_Loops = 1; 
Num_Duals = K + N;

W_nodes = cell(N, 1);
Mu_nodes = cell(N, 1);
for i = 1:N, W_nodes{i} = W_init; Mu_nodes{i} = zeros(Num_Duals, 1); end

dist_power_history = [];
cvx_power_history = [];
total_ppd_iterations = 0; 

fprintf('Starting Optimization with CVX Comparison...\n');

for ccp = 1:max_ccp_iter
    CCP_Weights = wheith(N);
    W_tensor_ccp = cat(3, W_nodes{:});
    W_flat_ccp   = reshape(W_tensor_ccp, [], N);
    W_flat_ccp_new = W_flat_ccp * CCP_Weights.';
    W_prev_ccp = squeeze(num2cell(reshape(W_flat_ccp_new, N, M, N), [1 2]));
    
    W_prev_mat = zeros(N, M);
    for n=1:N, W_prev_mat(n,:) = W_prev_ccp{n}(n,:); end 
    
    for k = 1:max_ppd_iter
        total_ppd_iterations = total_ppd_iterations + 1;
        step_size = 10 / ((1 + k) * ccp);
        
        V_W = W_nodes; V_Mu = Mu_nodes;
        for c_round = 1:Consensus_Loops
            Current_Weights = wheith(N);
            
            W_tensor = cat(3, V_W{:});
            W_flat = reshape(W_tensor, [], N);
            W_flat_new = W_flat * Current_Weights.';
            V_W = squeeze(num2cell(reshape(W_flat_new, N, M, N), [1 2]));
            
            Mu_matrix = [V_Mu{:}];
            Mu_matrix_new = Mu_matrix * Current_Weights.';
            V_Mu = num2cell(Mu_matrix_new, 1)';
        end
        
        W_next = cell(N,1); 
        
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
            if grad_norm > 100
                D_total = D_total * (100/grad_norm);
            end
            
            W_temp = vx - step_size * D_total;
            if norm(W_temp(i,:)) > sqrt(P_vec(i)), W_temp(i,:) = W_temp(i,:) * (sqrt(P_vec(i))/norm(W_temp(i,:))); end
            W_next{i} = W_temp;
        end
        
        Mu_next = cell(N,1);
        for i = 1:N
            vx = V_W{i}; vmu = V_Mu{i}; w_t = W_prev_ccp{i};
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
        
        W_nodes = W_next; Mu_nodes = Mu_next;
    end
    
    w_avg = zeros(N,M); 
    for n=1:N, w_avg = w_avg + W_nodes{n}; end; 
    w_avg = w_avg/N;
    curr_power = sum(sum(abs(w_avg).^2));
    dist_power_history = [dist_power_history; curr_power];
    
    cvx_begin quiet
        variable W_cvx(N, M) complex
        minimize( square_pos(norm(W_cvx, 'fro')) )
        subject to
            for n = 1:N
                sum_square_abs(W_cvx(n,:)) <= P_vec(n);
            end
            
            for k = 1:K
                interf_cvx = sigma2;
                for j = 1:M
                    if j ~= k
                        interf_cvx = interf_cvx + square_abs(H(:,k)' * W_cvx(:,j));
                    end
                end
                
                h_wk_prev = H(:,k)' * W_prev_mat(:,k);
                h_wk_curr = H(:,k)' * W_cvx(:,k);
                sig_lin_cvx = 2 * real(conj(h_wk_prev) * h_wk_curr) - abs(h_wk_prev)^2;
                
                gamma_vec(k) * interf_cvx <= sig_lin_cvx;
            end
    cvx_end
    
    if strcmpi(cvx_status, 'Solved')
        cvx_pow = sum(sum(abs(W_cvx).^2));
        cvx_power_history = [cvx_power_history; cvx_pow];
    else
        cvx_power_history = [cvx_power_history; NaN];
    end
    
    fprintf('CCP Iter %d: Dist Power = %.4f | CVX Power = %.4f\n', ccp, curr_power, cvx_power_history(end));
end

fprintf('\nTotal PPD Inner Iterations: %d\n', total_ppd_iterations);

figure;
semilogy(dist_power_history, '-s', 'LineWidth', 2, 'DisplayName', 'Distributed PPD'); 
hold on;
semilogy(cvx_power_history, '-o', 'LineWidth', 2, 'DisplayName', 'Centralized CVX');
grid on;
legend;
title('Power Minimization: Distributed vs Centralized'); 
ylabel('Power (Watts)'); xlabel('CCP Iteration');

fprintf('\n==============================================\n');
fprintf('       FINAL RESULT VERIFICATION\n');
fprintf('==============================================\n');

w_final = zeros(N, M);
for n = 1:N, w_final = w_final + W_nodes{n}; end
w_final = w_final / N;

fprintf('\n--- Power Constraint Check ---\n');
per_antenna_power = sum(abs(w_final).^2, 2);
tol_check = 1e-4;
sat_power = all(per_antenna_power <= P_vec + tol_check);

for n=1:N
    fprintf('Antenna %d: Power = %.4f W, Limit = %.4f W', n, per_antenna_power(n), P_vec(n));
    if per_antenna_power(n) <= P_vec(n) + tol_check
        fprintf(' [OK]\n'); 
    else
        fprintf(' [VIOLATED]\n'); 
    end
end

if sat_power
    fprintf('>> Result: All Power Constraints SATISFIED.\n');
else
    fprintf('>> Result: Some Power Constraints VIOLATED.\n');
end

fprintf('\n--- SINR Constraint Check ---\n');
sinr_vals = zeros(K,1);
sat_sinr = true;

for k=1:K
    sig = abs(H(:,k)' * w_final(:,k))^2;
    interf = 0;
    for j=1:M
        if j ~= k
            interf = interf + abs(H(:,k)' * w_final(:,j))^2;
        end
    end
    
    sinr_vals(k) = sig / (sigma2 + interf);
    
    fprintf('User %d: Target = %.2f, Achieved = %.4f ', k, gamma_vec(k), sinr_vals(k));
    
    if sinr_vals(k) >= gamma_vec(k) - 1e-4
        fprintf('[OK]\n'); 
    else
        fprintf('[VIOLATED]\n'); 
        sat_sinr = false; 
    end
end

if sat_sinr
    fprintf('>> Result: All SINR Constraints SATISFIED.\n');
else
    fprintf('>> Result: Some SINR Constraints VIOLATED.\n');
end

function W = wheith(N)
     W = (1/N) * ones(N,N);
end
