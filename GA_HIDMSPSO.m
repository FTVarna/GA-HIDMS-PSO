% -------------------------------------------------------------------------             
% Genetic Algorithm Assisted HIDMS-PSO: A New Hybrid Algorithm for Global Optimisation  
%
% Implemented by Fevzi Tugrul Varna - University of Sussex                              
% Cite as: ----------------------------------------------------------------
% F. T. Varna and P. Husbands, "Genetic Algorithm Assisted HIDMS-PSO: A New Hybrid 
% Algorithm for Global Optimisation," 2021 IEEE Congress on Evolutionary Computation
% (CEC), Kraków, Poland, 2021, pp. 1304-1311, doi: 10.1109/CEC45853.2021.9504852.       
% -------------------------------------------------------------------------             
%% expected inputs: fId,n,d,range where fId=function no., n=swarm size, d=dimension, range=lower and upper bounds
%% e.g. GA_HIDMSPSO(fhd,1,40,10,[-100 100])

function [fmin] = GA_HIDMSPSO(fhd,fId,n,d,range)
if rem(n,4)~=0, error("** Input Error: Swarm population must be divisible by 4 **"), end
rand('seed',sum(100*clock));

fevalCount=0;
LB=range(1);
UB=range(2);
Fmax=10^4*d;                %maximum number function evaluations
Tmax=Fmax/n;                %maximum number of iterations
%{
if d==10, Tmax=Tmax-289;
elseif d==30, Tmax=Tmax-900;
elseif d==50, Tmax=Tmax-1500;
elseif d==100, Tmax=Tmax-3000;
end
%}
ShowProgress=false;

%% Parameters of HIDMS-PSO
w1 = 0.99+(0.2-0.99)*(1./(1+exp(-5*(2*(1:Tmax)/Tmax-1))));      %nonlinear decrease inertia weight - Sigmoid function
c1 = 2.5-(1:Tmax)*2/Tmax;                                       %personal acceleration coefficient
c2 = 0.5+(1:Tmax)*2/Tmax;                                       %social acceleration coefficient

COM = false;                                                    %Communication model enabled/disabled
alpha_min = Tmax*0.01;
alpha_max = Tmax*0.1;
alpha = alpha_max;                                              %initial alpha value, determines units' reshape interval
UPn = 4;                            %unit pop size (constant)
U_n = (n/2)/UPn;                    %number of units (constant)
U = reshape(randperm(n/2),U_n,UPn); %units (U_n-by-UPn matrix)
[master,slave1,slave2,slave3] = feval(@(x) x{:}, num2cell([1,2,3,4])); %unit members' codes (constant)

%Velocity clamp
MaxV = 0.15*(UB-LB);
MinV = -MaxV;

GA = false;         %controls the switch between HIDMS-PSO and GA
PSO_phase = 100;    %number of iterations HIDMS-PSO will run before switching back to GA
GA_phase = 50;      %number of iterations GA will run before switching back to HIDMS-PSO

%% Initialisation
V = zeros(n,d);           %initial velocities
X = unifrnd(LB,UB,[n,d]); %initial positions
PX = X;                   %initial pbest positions
F = feval(fhd,X',fId);    %function evaluation
PF = F;                   %initial pbest cost
GX = [];                  %gbest solution vector
GF = inf;                 %gbest cost

%update gbest
for i=1:n
    if PF(i)<GF, GF=PF(i); GX=PX(i,:); end
end

%% Main Loop of GA-HIDMS-PSO
for t=1:Tmax

    %control the switch between GA and HIDMS-PSO
    if t<=Tmax*0.9
        if mod(t,PSO_phase)==0
            GA = true;
        end
    end

    %reshape units
    if mod(t,round(alpha))==0
        [~,idx] = sort(rand(U_n,UPn));
        U = U(sub2ind([U_n,UPn],idx,ones(U_n,1)*(1:UPn)));
    end

    for i=1:n
        if F(i) >= mean(F), w = w1(t) + 0.15; if w > 0.99,  w = 0.99;end
        else, w = w1(t) - 0.15; if w < 0.20,  w = 0.20;end
        end

        if t <= Tmax*0.9
            if ~isempty(find(U==i))                                     %if agent belongs to the heterogeneous subpop
                if randi([0 1]) == 0                                      %inward-oriented movement
                    if ~isempty(find(U(:,master)==i))                   %agent is master
                        behaviour = randi([1 3]);
                        [uId,~] = find(U(:,master)==i);                 %unit id of the ith (master) particle
                        if behaviour==1                                 %move towards the most dissimilar slave
                            sList = U(uId,slave1:slave3);               %get slaves of the master
                            similarities = zeros(1,length(sList));      %
                            for ii=1:length(sList)                      %calculate similarities between master and slave particles
                                similarities(ii) = immse(PF(i),PF(sList(ii))); %immse is used instead of mae and mse for faster performance
                            end
                            [~,dsId] = max(similarities);               %find the most dissimilar agent
                            dsId = sList(dsId);                         %idx of the dissimilar slave
                            V(i,:) = w*V(i,:)+c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(PX(dsId,:) - X(i,:));
                        elseif behaviour == 2                           %move towards the best slave
                            sList = U(uId,slave1:slave3);               %idx of slaves in the unit
                            slave_costs = [F(sList(1)) F(sList(2)) F(sList(3))];
                            [~,bsId] = min(slave_costs);                %best slave's idx
                            V(i,:) = w*V(i,:)+c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(PX(sList(bsId),:) - X(i,:));
                        elseif behaviour == 3                           %move towards average of slaves
                            sList = U(uId,slave1:slave3);               %get slaves of the master
                            slaves_pos = [X(sList(1),:); PX(sList(2),:); X(sList(3),:);];
                            V(i,:) = w*V(i,:)+c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(mean(slaves_pos) - X(i,:));
                        end
                    else                                                %agent is slave, move towards the master particle
                        [uId,~] = find(U(:,slave1:slave3)==i);          %find the unit particle belongs to
                        V(i,:) = w*V(i,:)+c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(PX(U(uId,master),:) - X(i,:));
                    end
                else	%outward-oriented movement
                    if ~isempty(find(U(:,master)==i))                   %agent is master
                        behaviour = randi([1 3]);                       %randomly selected behaviour
                        if behaviour==1                                 %if 1, move towards the avg pos of another unit
                            rndU = randi([1,U_n],1,1);                  %select a unit randomly
                            sList = U(rndU,slave1:slave3);              %get slaves of a random unit
                            uX = [PX(U(rndU,master),:); X(sList(1),:); PX(sList(2),:); X(sList(3),:)]; %all positions of the unit
                            V(i,:) = w*V(i,:)+c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(mean(uX) - X(i,:));
                        elseif behaviour==2                             %move towards the master of another unit
                            V(i,:) = w*V(i,:)+c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(PX(U(randi([1 U_n]),master),:) - X(i,:));
                        elseif behaviour==3                             %move towards avg of self unit and master of another unit
                            sList = U(find(U(:,master)==i),slave1:slave3);  %get all slaves of the unit
                            sList(4) = U(randi([1 U_n]),master);        %add a master particle from a random unit
                            V(i,:) = w*V(i,:)+c1(t)*rand([1 d]).*(mean(X(sList,:)) - X(i,:)) + c2(t)*rand([1 d]).*(PX(U(randi([1 U_n]),master),:) - X(i,:));
                        end
                    else                                                %agent is slave
                        [~,sType] = find(U==i);                         %find self slave type
                        Slist = U(:,sType);                             %get list of all slaves of the same type
                        [selfId,~] = find(Slist==i);
                        Slist(selfId) = [];                             %remove self from the list
                        rndSlave = randperm(length(Slist));             %shuffle the list
                        rndSlave = Slist(rndSlave(1));                  %select the first one
                        V(i,:) = w*V(i,:)+c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(X(rndSlave,:) - X(i,:));
                    end
                end
            else                                                        %velocity update for particles in the homogeneous subpop
                V(i,:) = w*V(i,:)+c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(GX - X(i,:));
            end
        else                                                            %final phase of the search process (exploitation)
            V(i,:) = w*V(i,:)+c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(GX - X(i,:));
        end

        V(i,:) = max(V(i,:),MinV); V(i,:) = min(V(i,:),MaxV);           %velocity clamp
        X(i,:) = X(i,:) + V(i,:);                                       %update position
        X(i,:) = max(X(i,:), LB); X(i,:) = min(X(i,:), UB);             %apply lower and upper bound limits

        %particle communication
        if COM==true
            if ~isempty(find(U==i))                                     %if particle belongs to the heterogeneous subpop
                [~,sIdx] = find(U==i);                                  %find the slave type of the ith particle
                if sIdx == slave1 || sIdx == slave2 || sIdx == slave3   %if ith agent is a slave
                    sList = U(:,sIdx);                                  %pool of same type of slaves from all units
                    rndSlave = randperm(length(sList),1);               %select a random slave from the pool
                    %positional info exchange between the ith particle and a random slave of the same type
                    if PF(i) < PF(sList(rndSlave)), PF(sList(rndSlave)) = PF(i); PX(sList(rndSlave),:) = PX(i,:);
                    else, PF(i) = PF(sList(rndSlave)); PX(i,:) = PX(sList(rndSlave),:);
                    end
                end
            end
        end
    end
    
    F = feval(fhd,X',fId);  %function evaluation
    fevalCount = fevalCount + n;

    for j=1:n
        if F(j) < PF(j), PF(j) = F(j); PX(j,:) = X(j,:); end  %update pbests
        if PF(j) < GF, GF = PF(j); GX = PX(j,:); end          %update gbest
    end

    alpha = round(alpha_max-(alpha_max-alpha_min)*t/Tmax);

    if GA==true
        GA_pop_ids = [randi([1 n],1,n/2)];      %randomly select half of the population from both homogeneous and heterogeneous populations
        GA_pop = X(GA_pop_ids,:);               %create a new population for GA
        [new_X,new_F,new_best,new_GF,fevalCount] = GeneticAlgorithm(fhd,fId,n/2,[LB UB],d,GA_pop,GA_phase,fevalCount);
        X(GA_pop_ids,:) = new_X;                %update particles with returned positions from GA
        F(GA_pop_ids) = new_F;                  %update particle fitness with returned fitness from GA

        %update gbest position and cost
        if new_GF < GF
            GX = new_best.Position;
            GF = new_GF;
        end
        GA = false;
    end

    if ShowProgress==true, disp(['Iteration '   num2str(t)  ' | Best cost = '  num2str(GF)]); end
end
fmin=GF;
end