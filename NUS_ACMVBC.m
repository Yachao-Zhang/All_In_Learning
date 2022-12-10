clear;  clear memory; clc;
addpath('./NUS')
addpath('./Utility')
addpath('./twist');

load NUS
clear categories feanames lenSmp

% Nonlinear anchor feature embedding
viewNum = size(X,2);
AnchorNum=1000;
load NUSAnchor
fprintf('Nonlinear Anchor Embedding...\n');
for it = 1:viewNum
    dist = EuDist2(X{it},Anchor{it},0);
    sigma = mean(min(dist,[],2).^0.5)*2;
    feaVec = exp(-dist/(2*sigma*sigma));
    X{it} = bsxfun(@minus, feaVec', mean(feaVec',2));% Centered data 
end
clear feaVec dist sigma dist Anchor it

%------------Initializing parameters--------------
MaxIter = 5;       %
innerMax = 10;
r = 6;              % r is the power of alpha_i
L = 128;            % Hashing code length
rhob = 1e-4; max_rhob = 10e10; pho_rhob = 8;%5
alph=1;%
gamma = 0.035;        % Hyper-para gamma 
lambda = 2e-5;   % Hyper-para lambda 
betaa=1e-6;

N = size(X{1},2);
rand('seed',100);
sel_sample = X{1}(:,randsample(N, AnchorNum),:); 
[pcaW, ~] = eigs(cov(sel_sample'), L);   
B = sign(pcaW'*X{1});  

n_cluster = numel(unique(Y));
alpha = ones(viewNum,1) / viewNum; 
U = cell(1,viewNum);

rand('seed',500); 
C = B(:,randsample(N, n_cluster)); 
HamDist = 0.5*(L - B'*C);
[~,ind] = min(HamDist,[],2);
G = sparse(ind,1:N,1,n_cluster,N,N);
G = full(G);
CG = C*G;

XXT = cell(1,viewNum);
for view = 1:viewNum
    XXT{view} = X{view}*X{view}';
end
clear HamDist ind initInd n_randm pcaW sel_sample view

sX=[L,AnchorNum,viewNum];
for k=1:viewNum        
     TG{1}{k}=double(zeros(L,AnchorNum));% low-rank tensor    
     TW{1}{k}=double(zeros(L,AnchorNum)); % Lagrange multiplier for W        
end

intr=(viewNum-1).*eye(AnchorNum);%
inter=eye(AnchorNum);  
CC=[intr,-inter,-inter,-inter,-inter;
   -inter,intr,-inter,-inter,-inter;
   -inter,-inter,intr,-inter,-inter;
   -inter,-inter,-inter,intr,-inter;
   -inter,-inter,-inter,-inter,intr;
   ];
XX=[X{1}',X{2}',X{3}',X{4}',X{5}'];
XXXT=[XXT{1},zeros(size(XXT{1})),zeros(size(XXT{1})),zeros(size(XXT{1})),zeros(size(XXT{1}));
       zeros(size(XXT{1})),XXT{2},zeros(size(XXT{1})),zeros(size(XXT{1})),zeros(size(XXT{1}));
       zeros(size(XXT{1})),zeros(size(XXT{1})),XXT{3},zeros(size(XXT{1})),zeros(size(XXT{1}));
       zeros(size(XXT{1})),zeros(size(XXT{1})),zeros(size(XXT{1})),XXT{4},zeros(size(XXT{1}));
       zeros(size(XXT{1})),zeros(size(XXT{1})),zeros(size(XXT{1})),zeros(size(XXT{1})),XXT{5};
       ];
clear inter intr
disp('----------The proposed method (multi-view)----------');
tic
% profile on -memory
for iter = 1:MaxIter

    fprintf('The %d-th iteration...\n',iter);
    %---------Update Ui--------------
    %% 
    alpha_r = alpha.^r;
    UX = zeros(L,N);
    TWW{1}=[TW{1}{1},TW{1}{2},TW{1}{3},TW{1}{4},TW{1}{5}];
    TGG{1}=[TG{1}{1},TG{1}{2},TG{1}{3},TG{1}{4},TG{1}{5}];
    UU=(2*B*XX+rhob*TGG{1}-TWW{1})/((1-gamma)*XXXT+rhob*eye(size(XX,2))+betaa.*CC);%   

    %% %---------Update B--------------
    for v=1:viewNum
        U{v}=UU(:,1+(v-1)*AnchorNum:AnchorNum*v); 
        UX   = UX+alpha_r(v)*U{v}*X{v};
    end
    B = sign(UX+lambda*CG);B(B==0) = -1;
    %% %----------Update TG and TW ---------------
   tensor_u=[];
   tensor_w=[];
   tensor_u=[tensor_u(:);UU(:)];
   tensor_w=[tensor_w(:);TWW{1}(:)];            
   [tensor_g, objV] = wshrinkObj(tensor_u + 1/rhob*tensor_w,alph/rhob,sX,0,3);
   tensor_g=reshape(tensor_g,sX);%

   for v=1:viewNum
       TG{1}{v} = tensor_g(:,:,v);
       TW{1}{v}=TW{1}{v}+rhob*(U{v}-TG{1}{v});
   end
%    fprintf('    norm_z %7.10f   \n ',norm_Z);
   rhob = min(rhob*pho_rhob, max_rhob);
    %----------------------------------
    %% 
    %---------Update C and G--------------
    for iterInner = 1:innerMax
        % For simplicity, directly using DPLM here
        C = sign(B*G'); C(C==0) = 1;
        rho = .001; mu = .01; % Preferred for this dataset
        for iterIn = 1:3
            grad = -B*G' + rho*repmat(sum(C),L,1);
            C    = sign(C-1/mu*grad); C(C==0) = 1;
        end
        HamDist = 0.5*(L - B'*C); % Hamming distance referring to "Supervised Hashing with Kernels"
        [~,indx] = min(HamDist,[],2);
        G = sparse(indx,1:N,1,n_cluster,N,N);
    end
    CG = C*G;
    %% 
    h = zeros(viewNum,1);
    for view = 1:viewNum
        UCU=trace(UU(:,1+((view-1)*1000):1000*view)*CC(1+((view-1)*1000):1000*view,1+((view-1)*1000):1000*view)*UU(:,1+((view-1)*1000):1000*view)');
        h(view) = norm(B-U{view}*X{view},'fro')^2 -gamma*norm(U{view}*X{view},'fro')^2 + alph*norm(U{view},'fro')^2+betaa.*UCU;
    end
    H = bsxfun(@power,h, 1/(1-r));     % h = h.^(1/(1-r));
    alpha = bsxfun(@rdivide,H,sum(H)); % alpha = H./sum(H);
end
disp('----------Main Iteration Completed----------');
disp(['OUR-time£º',num2str(toc)]);
[~,pred_label] = max(G,[],1);
res_cluster = ClusteringMeasure(Y, pred_label);
fprintf('All view results: ACC = %.4f and NMI = %.4f, Purity = %.4f\n\n',res_cluster(1),res_cluster(2),res_cluster(3));