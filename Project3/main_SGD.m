clearvars, close all;
% proj3 problem3; Fusar, Galimberti

% input data
x1 = [0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7];
x2 = [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6];
label = [ones(1,5), zeros(1,5)];        % labels for training data

% plot training data
% we plot Class 1 (label 1) as blue circles and Class 0 as red circles
figure;
P_train = [x1; x2];
scatter(P_train(1, label==1), P_train(2, label==1), 'bo', 'filled','LineWidth',3);
hold on;
scatter(P_train(1, label==0), P_train(2, label==0), 'ro', 'filled','LineWidth',3);

% data to be predicted
P_test = [0.8 0 0.4;
          0.4 0.4 0.6];
scatter(P_test(1,:), P_test(2,:), 'k*','LineWidth',3);
legend("B","A","To be classified")
grid on
axis([-0.1 1.01 -0.1 1.01]);
title('Data');

% data feeding to the model
X = [x1; x2];                       % input data
label = [ones(1,5) zeros(1,5); zeros(1,5) ones(1,5)];
Y = label;                          % input Labels
samples_number = size(X, 2);        % number of samples
input_layer_size = size(X, 1);      % number of features (dimensions)
num_labels = size(Y, 1);            % number of labels

% construction of the fully connected net
layer_sizes = [input_layer_size, 2, 3, num_labels];
bias = cell(1, length(layer_sizes));                % initialize bias cell array
for i = 2:length(layer_sizes)
    bias{i} = randn(layer_sizes(i), 1);             % bias initialization
end
initial_weights = cell(1, length(layer_sizes)-1);   % initialize weights cell array
for i = 1:(length(layer_sizes)-1)
    initial_weights{i} = randInitializeWeights(layer_sizes(i), layer_sizes(i+1));
end

% training parameters
alfa = 5e-2;
maxIter = 1000000;
batch_size=1;
batch_accuracy_size=10;     % determines size of input on which compute the accuracy after i steps,
                            % necessary for bigger input data sizes
                            
% visualization and training process
loss = [];
count = [];
precisionT = [];
weights = initial_weights;
figure;
ax1 = subplot(2,1,1); ax1.YGrid = "on"; ax1.XGrid = "on";
ax2 = subplot(2,1,2); ax2.YGrid = "on"; ax2.XGrid = "on";
tic;
true_labels = Y';
for i = 1:maxIter
    randomIndices = randperm(samples_number, batch_size);
    [J, weights] = train(weights, bias, layer_sizes, X(:,randomIndices), Y(:,randomIndices), alfa);
    if mod(i,25000) == 0
        loss = [loss; J];
        count = [count; i];
        plot(ax1, count, loss, 'LineWidth', 2); ax1.YGrid = "on"; ax1.XGrid = "on";
        title(ax1, 'Loss Evolution');
        xlabel(ax1, 'Iterations');
        ylabel(ax1, 'Loss function');

        % precision calculation 
        randomIndices = randperm(samples_number, batch_accuracy_size);
        pred = predict(weights, bias, X(:,randomIndices), layer_sizes);
        true_labels = Y(:,randomIndices)'
        difference = pred - true_labels;
        difference = pred - true_labels;
        norm2_per_row = sqrt(sum(difference.^2, 2));
        precision = mean(norm2_per_row);
        precisionT = [precisionT; 1-precision];
        plot(ax2, count, precisionT, 'LineWidth', 2); ax2.YGrid = "on"; ax2.XGrid = "on";
        title(ax2, 'Accuracy Evolution');
        xlabel(ax2, 'Iterations');
        ylabel(ax2, 'Accuracy');
        display(pred);
        disp(['Iteration #: ' num2str(i) ' / ' num2str(maxIter) ' | Loss J: ' num2str(J) ' | Accuracy: ' num2str(1-precision)]);
        drawnow();
    end
end
finT = toc;
disp(['Time spent on training the net: ' num2str(finT) ' seconds']);

% Prediction Plane Display
N = 100;
[x, y] = meshgrid(linspace(0, 1, N), linspace(0, 1, N));
P = [x(:)'; y(:)'];         % P is now a matrix where each column is a point [x;y]
plane_prediction = predict_label(weights, bias, P, layer_sizes);
figure
scatter(P(1, plane_prediction(1, :) == 1), P(2, plane_prediction(1, :) == 1), 'bo', 'filled', 'LineWidth', 0.0000001);
hold on;
scatter(P(1, plane_prediction(1, :) == 0), P(2, plane_prediction(1, :) == 0), 'ro', 'filled', 'LineWidth', 0.0000001);
scatter(P_test(1,:), P_test(2,:), 'k*','LineWidth',3);
title('Predictions on 100x100 Grid');       % corrected grid size in title
xlabel('X');
ylabel('Y');
legend('Class B', 'Class A', 'To be Classified');
hold off;

% comparison for varying alpha
maxIter = 100000;

true_labels=Y';
layer_sizes = [input_layer_size, 2, 3, num_labels];
bias1 = cell(1, length(layer_sizes));                % initialize bias cell array
bias2 = cell(1, length(layer_sizes));
bias3 = cell(1, length(layer_sizes));
loss1 = []; loss2 = []; loss3 = [];
count1 = []; count2 = []; count3 = [];
precisionT1 = []; precisionT2 = []; precisionT3 = [];
for i = 2:length(layer_sizes)
    bias1{i} = randn(layer_sizes(i), 1);             % bias initialization
    bias2{i} = randn(layer_sizes(i), 1);
    bias3{i} = randn(layer_sizes(i), 1);
end
initial_weights1 = cell(1, length(layer_sizes)-1);   % initialize weights cell array
initial_weights2 = cell(1, length(layer_sizes)-1);
initial_weights3 = cell(1, length(layer_sizes)-1);
for i = 1:(length(layer_sizes)-1)
    weights1{i} = randInitializeWeights(layer_sizes(i), layer_sizes(i+1));
    weights2{i} = randInitializeWeights(layer_sizes(i), layer_sizes(i+1));
    weights3{i} = randInitializeWeights(layer_sizes(i), layer_sizes(i+1));
end

for i = 1:maxIter
    randomIndices = randperm(samples_number, batch_size);
    [J1, weights1] = train(weights1, bias1, layer_sizes, X(:,randomIndices), Y(:,randomIndices), alfa/20);
    [J2, weights2] = train(weights2, bias2, layer_sizes, X(:,randomIndices), Y(:,randomIndices), alfa);
    [J3, weights3] = train(weights3, bias3, layer_sizes, X(:,randomIndices), Y(:,randomIndices), alfa*20);
    
    if mod(i,25) == 0
        loss1 = [loss1; J1]; loss2 = [loss2; J2]; loss3 = [loss3; J3];
        count1 = [count1; i]; count2 = [count2; i]; count3 = [count3; i];
        plot(ax1, count, loss, 'LineWidth', 2); ax1.YGrid = "on"; ax1.XGrid = "on";
        title(ax1, 'Loss Evolution');
        xlabel(ax1, 'Iterations');
        ylabel(ax1, 'Loss function');

        % precision calculation
        pred1 = predict(weights1, bias1, X, layer_sizes);
        pred2 = predict(weights2, bias2, X, layer_sizes);
        pred3 = predict(weights3, bias3, X, layer_sizes);
        difference1 = pred1 - true_labels;
        difference2 = pred2 - true_labels;
        difference3 = pred3 - true_labels;
        norm2_per_row1 = sqrt(sum(difference1.^2, 2));
        norm2_per_row2 = sqrt(sum(difference2.^2, 2));
        norm2_per_row3 = sqrt(sum(difference3.^2, 2));
        precision1 = mean(norm2_per_row1);
        precision2 = mean(norm2_per_row2);
        precision3 = mean(norm2_per_row3);
        precisionT1 = [precisionT1; 1-precision1];
        precisionT2 = [precisionT2; 1-precision2];
        precisionT3 = [precisionT3; 1-precision3];
    end
end

figure
subplot(1,3,1)
plot( count1, precisionT1, 'LineWidth', 2); ax2.YGrid = "on"; ax2.XGrid = "on";
title(ax2, 'Accuracy Evolution');
xlabel(ax2, 'Iterations');
ylabel(ax2, 'Accuracy');
title('alpha1=alpha/20');
subplot(1,3,2)
plot( count2, precisionT2, 'LineWidth', 2); ax2.YGrid = "on"; ax2.XGrid = "on";
title(ax2, 'Accuracy Evolution');
xlabel(ax2, 'Iterations');
ylabel(ax2, 'Accuracy');
title('alpha2=alpha');
subplot(1,3,3)
plot( count3, precisionT3, 'LineWidth', 2); ax2.YGrid = "on"; ax2.XGrid = "on";
title(ax2, 'Accuracy Evolution');
xlabel(ax2, 'Iterations');
ylabel(ax2, 'Accuracy');
title('alpha3=alpha*20');

load('weights.mat', 'weights');
load('bias.mat', 'bias');
xt1=[0.8; 0.4];         % expecting category B
xt2=[0; 0.4];           % expenting category A
xt3=[0.4; 0.6]; 
xt4=[0.444; 0.6];       % uncertain case
v=[xt1 xt2 xt3 xt4];
prediction=predict(weights,bias,v,layer_sizes);
display(prediction);

% function
function [J, weights] = train(weights, bias, layer_sizes, X, Y, alpha)

% initialize variables
m = size(X, 2);     % number of training examples
num_layers = length(layer_sizes);
J = 0;              % cost function

% loop over all examples for SGD
for i = 1:m
    % forward propagation
    a = cell(num_layers, 1);        % activations
    z = cell(num_layers-1, 1);      % weighted inputs

    % input layer
    a{1} = X(:, i);
    
    % forward pass
    for layer = 2:num_layers
        z{layer-1} = weights{layer-1} * a{layer-1} + bias{layer};
        a{layer} = sigmoid(z{layer-1});
    end
 
    % compute cost using MSE
    J = J + 0.5 * sum((a{end} - Y(:, i)).^2); % Accumulate squared error
    
    % backpropagation
    delta = cell(num_layers, 1);
    delta{end} = a{end} - Y(:, i);

    for layer = (num_layers-1):-1:2
        delta{layer} = (weights{layer}' * delta{layer+1}) .* sigmoidGradient(z{layer-1});
    end
    
    % gradient descent step for weights and biases
    for layer = 1:(num_layers-1)
        weights{layer} = weights{layer} - alpha * (delta{layer+1} * a{layer}') / m;
        bias{layer+1} = bias{layer+1} - alpha * mean(delta{layer+1}, 2);
    end
end

J = J / m;

end

% helper functions
function g = sigmoid(z)

g = 1.0 ./ (1.0 + exp(-z));

end

function g = sigmoidGradient(z)

g = sigmoid(z) .* (1 - sigmoid(z));

end

function W = randInitializeWeights(L_in, L_out)

epsilon_init = sqrt(6) / sqrt(L_in + L_out);
W = rand(L_out, L_in) * 2 * epsilon_init - epsilon_init;

end

function pred = predict(weights, bias, X, layer_sizes)

m = size(X, 2);
a = X;
    
for layer = 2:length(layer_sizes)
    z = weights{layer-1} * a + bias{layer};
    a = sigmoid(z);
end
pred = a';

end

function pred = predict_label(weights, bias, X, layer_sizes)

m = size(X, 2);         % number of samples
a = X;
    
for layer = 2:length(layer_sizes)
    z = weights{layer-1} * a + bias{layer};
    a = sigmoid(z);
end

% 'a' now contains the activations of the last layer for each sample
% convert the activations to predictions in 2D format
pred = zeros(layer_sizes(end), m);      % initialize prediction matrix

% for each sample, set the class with highest activation to 1
for i = 1:m
    [~, max_index] = max(a(:,i));       % find index of max activation for this sample
    pred(max_index, i) = 1;
end

end
