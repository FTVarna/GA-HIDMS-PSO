function [X,F,ga_BestSol,fmin,fevalCount] = GeneticAlgorithm(fhd,funcId,nPop,bnds,d,X,Tmax,fevalCount)
LB = bnds(1);
UB = bnds(2);

F=zeros(1,nPop); %costs of returning particles
%% GA Parameters
pc=0.7;                      % Crossover Percentage
nc=2*round(pc*nPop/2);       % Number of Offsprings (also Parnets)
gamma=0.4;                   % Extra Range Factor for Crossover
pm=0.3;                      % Mutation Percentage
nm=round(pm*nPop);           % Number of Mutants
mu=0.1;                      % Mutation Rate
TournamentSize=3;            % Tournamnet Size

%% GA Initialization
empty_individual.Position=[];
empty_individual.Cost=[];
pop=repmat(empty_individual,nPop,1);
fmin = inf;

for i=1:nPop
    pop(i).Position=X(i,:);
    pop(i).Cost=feval(fhd,pop(i).Position',funcId);
end

% Sort Population
costs=[pop.Cost];
[costs, SortOrder]=sort(costs);
pop=pop(SortOrder);
ga_BestSol=pop(1);           % Store Best Solution
ga_BestCost=zeros(Tmax,1);     % Array to Hold Best Cost Values
WorstCost=pop(end).Cost;     % Store Cost

saved_sol = zeros(nPop,d); %solution saved by the GA to be passed on to PSO
saved_cost = inf*ones(1,nPop);%cost saved by the GA to be passed on to PSO

%% Genetic Algorithm Main Loop
for it=1:Tmax
    
    %check criteria and save solution
    for kk=1:nPop
        if pop(kk).Cost < saved_cost(kk)
            %save solutions to feed PSO
            saved_sol(kk,:) = pop(kk).Position;
            saved_cost(kk) = pop(kk).Cost;
        end
    end
    
    % Crossover
    popc=repmat(empty_individual,nc/2,2);
    for k=1:nc/2
        % Select Parents Indices
        i1=TournamentSelection(pop,TournamentSize);
        i2=TournamentSelection(pop,TournamentSize);
        
        % Select Parents
        p1=pop(i1);
        p2=pop(i2);
        
        % Apply Crossover
        [popc(k,1).Position,popc(k,2).Position]=Crossover(p1.Position,p2.Position,gamma,bnds);
       
        % Evaluate Offsprings
        popc(k,1).Cost=feval(fhd,popc(k,1).Position',funcId);
        popc(k,2).Cost=feval(fhd,popc(k,2).Position',funcId);
        fevalCount=fevalCount+1;
    end
    popc=popc(:);
    % Mutation
    popm=repmat(empty_individual,nm,1);
    for k=1:nm
        % Select Parent
        i=randi([1 nPop]);
        p=pop(i);
        popm(k).Position=Mutate(p.Position,mu,LB,UB);    % Apply Mutation
        popm(k).Cost=feval(fhd,popm(k).Position',funcId);   % Evaluate Mutant
        fevalCount=fevalCount+1;
    end
    
    % Create Merged Population
    pop=[pop
        popc
        popm]; %#ok

    % Sort Population
    costs=[pop.Cost];
    [costs, SortOrder]=sort(costs);
    pop=pop(SortOrder);
    WorstCost=max(costs);  % Update Worst Cost
    % Truncation
    pop=pop(1:nPop);
    costs=costs(1:nPop);
    % Store Best Solution Ever Found
    ga_BestSol=pop(1);
    % Store Best Cost Ever Found
    ga_BestCost(it)=ga_BestSol.Cost;
    fmin=ga_BestSol.Cost;
end

for ii=1:nPop
    X(ii,:) = pop(ii).Position;
    F(ii) = pop(ii).Cost;
end


function y=Mutate(x,mu,VarMin,VarMax)
nVar=numel(x);
nmu=ceil(mu*nVar);
j=randsample(nVar,nmu);
j=j'; %vertical vector j causes error
sigma=0.1*(VarMax-VarMin);
y=x;
y(j)=x(j)+sigma*randn(size(j));
y=max(y,VarMin);
y=min(y,VarMax);

function i=TournamentSelection(pop,m)
nPop=numel(pop);
S=randsample(nPop,m);
spop=pop(S);
scosts=[spop.Cost];
[~, j]=min(scosts);
i=S(j);

function [y1,y2]=Crossover(x1,x2,gamma,bnds)
%x1
%x2
alpha=unifrnd(-gamma,1+gamma,size(x1));
y1=alpha.*x1+(1-alpha).*x2;
y2=alpha.*x2+(1-alpha).*x1;
y1=max(y1,bnds(1));
y1=min(y1,bnds(2));
y2=max(y2,bnds(1));
y2=min(y2,bnds(2));