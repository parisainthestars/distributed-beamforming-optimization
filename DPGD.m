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

max_ccp_iter = 15;
max_gd_iter  = 3000;
rho_penalty  = 80;
Consensus_Loops = 1;

W_init = sqrt(P_watts/M) * (randn(N, M) + 1i * randn(N, M))/sqrt(2);
for i=1:N
    if norm(W_init(i,:))^2 > P_vec(i)
        W_init(i,:) = W_init(i,:) * sqrt(P_vec(i))/norm(W_init(i,:));
    end
end

W_nodes = cell(N, 1);
for i = 1:N, W_nodes{i} = W_init; end

hist_power = [];
hist_cvx = [];

fprintf('Starting Distributed Projected Gradient Descent...\n');

for ccp = 1:max_ccp_iter
    
    W_tensor = cat(3, W_nodes{:});
    W_avg = mean(W_tensor, 3);
    W_prev_mat = W_avg; 
    
    for k = 1:max_gd_iter
        
        alpha_k = 10 / ((1 +  k)*(ccp)); 
        
        V_W = W_nodes;
        Weights = (1/N) * ones(N,N); 
        
        for c = 1:Consensus_Loops
            W_tensor = cat(3, V_W{:});
            W_flat = reshape(W_tensor, [], N); 
            W_flat_mixed = W_flat * Weights.'; 
            V_W = squeeze(num2cell(reshape(W_flat_mixed, N, M, N), [1 2]));
        end
        
        W_next = cell(N, 1);
        
        for i = 1:N
            W_local = V_W{i}; 
            
            Grad_Total = 2 * W_local; 
            
            for u = 1:K
                interf_power = sigma2;
                for j = 1:M
                    if j ~= u
                        interf_power = interf_power + abs(H(:,u)' * W_local(:,j))^2;
                    end
                end
                
                h_prev = H(:,u)' * W_prev_mat(:,u);
                h_curr = H(:,u)' * W_local(:,u);
                sig_lin = 2 * real(conj(h_prev) * h_curr) - abs(h_prev)^2;
                
                constraint_val = gamma_vec(u) * interf_power - sig_lin;
                
                if constraint_val > 0
                    penalty_factor = 2 * rho_penalty * constraint_val;
                    
                    for j = 1:M
                        if j ~= u
                            h_u = H(:,u);
                            Grad_Interf = 2 * h_u * (h_u' * W_local(:,j));
                            Grad_Total(:,j) = Grad_Total(:,j) + penalty_factor * gamma_vec(u) * Grad_Interf;
                        end
                    end
                    
                    h_u = H(:,u);
                    Grad_Signal = -2 * h_u * (h_u' * W_prev_mat(:,u));
                    Grad_Total(:,u) = Grad_Total(:,u) + penalty_factor * Grad_Signal;
                end
            end
            
            if norm(Grad_Total, 'fro') > 100
                 Grad_Total = Grad_Total * (100 / norm(Grad_Total, 'fro'));
            end
            
            W_temp = W_local - alpha_k * Grad_Total;
            
            for n_idx = 1:N
                row_pwr = norm(W_temp(n_idx,:))^2;
                if row_pwr > P_vec(n_idx)
                    scale = sqrt(P_vec(n_idx) / row_pwr);
                    W_temp(n_idx,:) = W_temp(n_idx,:) * scale;
                end
            end
            
            W_next{i} = W_temp;
        end
        W_nodes = W_next;
    end
    
    w_final_iter = W_nodes{1};
    curr_power = sum(sum(abs(w_final_iter).^2));
    hist_power = [hist_power; curr_power];
    
    cvx_begin quiet
        variable W_cvx(N, M) complex
        minimize( square_pos(norm(W_cvx, 'fro')) )
        subject to
            for n = 1:N, sum_square_abs(W_cvx(n,:)) <= P_vec(n); end
            for k = 1:K
                interf_cvx = sigma2;
                for j = 1:M, if j ~= k, interf_cvx = interf_cvx + square_abs(H(:,k)' * W_cvx(:,j)); end; end
                
                h_wk_prev = H(:,k)' * W_prev_mat(:,k);
                h_wk_curr = H(:,k)' * W_cvx(:,k);
                sig_lin_cvx = 2 * real(conj(h_wk_prev) * h_wk_curr) - abs(h_wk_prev)^2;
                
                gamma_vec(k) * interf_cvx <= sig_lin_cvx;
            end
    cvx_end
    if strcmp(cvx_status, 'Solved')
        hist_cvx = [hist_cvx; sum(sum(abs(W_cvx).^2))];
    else
        hist_cvx = [hist_cvx; NaN];
    end
    
    fprintf('CCP %2d: Distributed Power = %.4f | CVX Power = %.4f\n', ccp, curr_power, hist_cvx(end));
end

figure;
semilogy(hist_power, '-s', 'LineWidth', 2, 'DisplayName', 'Distributed Projected Gradient');
hold on;
semilogy(hist_cvx, '-o', 'LineWidth', 2, 'DisplayName', 'Centralized CVX');
grid on; legend;
xlabel('CCP Iteration'); ylabel('Total Power (Watts)');
title('Convergence of Distributed Projected Gradient Algorithm');

fprintf('\n==============================================\n');
fprintf('       FINAL RESULT VERIFICATION\n');
fprintf('==============================================\n');

w_final = zeros(N, M);
for n = 1:N, w_final = w_final + W_nodes{n}; end
w_final = w_final / N;

fprintf('\n--- Power Constraint Check ---\n');
per_antenna_power = sum(abs(w_final).^2, 2);
sat_power = all(per_antenna_power <= P_vec + 1e-4);
for n=1:N
    fprintf('Antenna %d: Power = %.4f W, Limit = %.4f W', n, per_antenna_power(n), P_vec(n));
    if per_antenna_power(n) <= P_vec(n) + 1e-4
        fprintf(' [OK]\n'); 
    else
        fprintf(' [VIOLATED]\n'); 
    end
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
    
    if sinr_vals(k) >= gamma_vec(k) * 0.99 
        fprintf('[OK]\n'); 
    else
        fprintf('[VIOLATED]\n'); 
        sat_sinr = false; 
    end
end

if sat_power && sat_sinr
    fprintf('\n>> SUCCESS: All constraints satisfied.\n');
else
    fprintf('\n>> WARNING: Some constraints are violated. Try increasing rho_penalty or max_gd_iter.\n');
end
