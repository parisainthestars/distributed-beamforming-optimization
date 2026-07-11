clc; close all; clear all;

max_ccp_iter = 30;
max_ppd_iter = 3000; 
Consensus_Loops = 1; 

scenarios = {
    'Fully Connected (DS)', ...
    'Ring Connected (Stoch)', ...
    'Line Connected (Stoch)', ...
    'Periodic (Switch 1 iter)', ...
    'Periodic (Switch 50 iter)', ...
    'Periodic (Switch 100 iter)', ...
    'Fully Connected (Stoch)', ...
    'Random Connected (Stoch)', ...
    'Periodic (Switch 500 iter)'
};

styles = {'-o', '-s', '-^', '--', '--', '--', '-*', '-+', ':'};
colors = {'k', 'b', 'g', 'r', 'm', 'c', 'y', [0.5 0.2 0.8], [1 0.5 0]};

all_power_histories = [];
final_solutions = cell(length(scenarios),1);

N = 4;

W_Full_DS = (1/N) * ones(N,N);

W_Ring = [1/3 1/3 0 1/3; 
          1/3 1/3 1/3 0; 
          0 1/3 1/3 1/3; 
          1/3 0 1/3 1/3];

W_Line = [1/2 1/2 0 0; 
          1/3 1/3 1/3 0; 
          0 1/3 1/3 1/3; 
          0 0 1/2 1/2];

W_State1 = [0.5 0 0.5 0; 
            0 0.5 0 0.5; 
            0.5 0 0.5 0; 
            0 0 0.5 0.5];

W_State2 = [0.5 0.5 0 0; 
            0.5 0.5 0 0; 
            0 0 0.5 0.5; 
            0 0 0.5 0.5];

rng(100); 
W_temp = rand(N,N);
W_Full_Stoch = W_temp ./ sum(W_temp, 2);

rng(200);
W_temp2 = rand(N,N);
W_Random_Stoch = W_temp2 ./ sum(W_temp2, 2);

for s = 1:length(scenarios)
    fprintf('\n==============================================\n');
    fprintf('Running Scenario %d: %s\n', s, scenarios{s});
    fprintf('==============================================\n');
    
    rng(0); 
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
    
    Num_Duals = K + N;
    W_nodes = cell(N, 1);
    Mu_nodes = cell(N, 1);
    for i = 1:N, W_nodes{i} = W_init; Mu_nodes{i} = zeros(Num_Duals, 1); end
    dist_power_history = [];
    
    for ccp = 1:max_ccp_iter
        
        W_tensor_ccp = cat(3, W_nodes{:});
        W_flat_ccp   = reshape(W_tensor_ccp, [], N);
        W_flat_ccp_new = W_flat_ccp * W_Full_DS.'; 
        W_prev_ccp = squeeze(num2cell(reshape(W_flat_ccp_new, N, M, N), [1 2]));
        
        for k = 1:max_ppd_iter
            step_size = 10 / ((1 + k) * ccp);
            
            switch s
                case 1 
                    Current_Weights = W_Full_DS;
                case 2 
                    Current_Weights = W_Ring;
                case 3 
                    Current_Weights = W_Line;
                case 4 
                    if mod(k-1, 2) < 1, Current_Weights = W_State1;
                    else, Current_Weights = W_State2; end
                case 5 
                    if mod(k-1, 100) < 50, Current_Weights = W_State1;
                    else, Current_Weights = W_State2; end
                case 6 
                    if mod(k-1, 200) < 100, Current_Weights = W_State1;
                    else, Current_Weights = W_State2; end
                case 7 
                    Current_Weights = W_Full_Stoch;
                case 8
                    Current_Weights = W_Random_Stoch;
                case 9
                    if mod(k-1, 1000) < 500, Current_Weights = W_State1;
                    else, Current_Weights = W_State2; end
            end
            
            V_W = W_nodes; V_Mu = Mu_nodes;
            
            for c_round = 1:Consensus_Loops
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
                
                W_temp_p = vx - step_size * D_total;
                if norm(W_temp_p(i,:)) > sqrt(P_vec(i)), W_temp_p(i,:) = W_temp_p(i,:) * (sqrt(P_vec(i))/norm(W_temp_p(i,:))); end
                W_next{i} = W_temp_p;
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
        fprintf('  CCP %d Done. Power: %.4f\n', ccp, curr_power);
    end
    all_power_histories = [all_power_histories, dist_power_history];
    final_solutions{s} = w_avg;
end

figure; hold on;
for s=1:length(scenarios)
    semilogy(all_power_histories(:,s), styles{s}, 'Color', colors{s}, 'LineWidth', 1.5, 'DisplayName', scenarios{s});
end
grid on;
legend('Location', 'Best');
xlabel('CCP Iteration');
ylabel('Total Power (Watts)');
title('Comparison of Network Topologies & Switching Frequencies');

fprintf('\n=======================================\n');
fprintf('     FINAL CONSTRAINT VERIFICATION     \n');
fprintf('=======================================\n');
fprintf('%-30s | %-15s | %-15s\n', 'Scenario', 'Power Violations', 'SINR Violations');
fprintf('----------------------------------------------------------------\n');
for s = 1:length(scenarios)
    w_final = final_solutions{s};
    
    p_viol = 0;
    for n=1:N
        if norm(w_final(n,:))^2 > P_vec(n) + 1e-4, p_viol = p_viol + 1; end
    end
    
    s_viol = 0;
    for k=1:K
        sig = abs(H(:,k)' * w_final(:,k))^2;
        interf = 0;
        for j=1:M, if j~=k, interf = interf + abs(H(:,k)' * w_final(:,j))^2; end; end
        sinr = sig / (sigma2 + interf);
        if sinr < gamma_vec(k) - 1e-3, s_viol = s_viol + 1; end
    end
    
    fprintf('%-30s | %-15d | %-15d\n', scenarios{s}, p_viol, s_viol);
end