function X = prior(d,N)

xmin = -1*ones(d,1);
xmax = 1*ones(d,1);

X = nan(d,N);

for i = 1:N
    X(:,i) = unifrnd(xmin,xmax);
end    

end