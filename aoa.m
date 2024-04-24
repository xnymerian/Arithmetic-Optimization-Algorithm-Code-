% Aritmetik Optimizasyon Algoritması fonksiyonu
function [best_fitness, convergence_curve] = arithmetic_optimization(dim, pop_size, max_iter, func_num)
    % Başlangıç popülasyonunu oluştur
    population = rand(pop_size, dim);
    
    % Değerlendirme
    fitness = benchmark_func(population, func_num);
    
    % En iyi bireyi bul
    [best_fitness, best_index] = min(fitness);
    best_solution = population(best_index, :);
    
    % Yakınsama dizisi
    convergence_curve = zeros(max_iter, 1);
    
    % İterasyon döngüsü
    for iter = 1:max_iter
        % Aritmetik operatörlerle yeni popülasyon oluştur
        new_population = zeros(pop_size, dim);
        for i = 1:pop_size
            r1 = randi(pop_size);
            r2 = randi(pop_size);
            r3 = randi(pop_size);
            
            new_population(i, :) = population(r1, :) + rand * (population(r2, :) - population(r3, :));
        end
        
        % Yeni popülasyonu değerlendir
        new_fitness = benchmark_func(new_population, func_num);
        
        % Birleştirme ve seçim
        combined_population = [population; new_population];
        combined_fitness = [fitness; new_fitness];
        
        [~, sorted_indices] = sort(combined_fitness);
        
        population = combined_population(sorted_indices(1:pop_size), :);
        fitness = combined_fitness(sorted_indices(1:pop_size));
        
        % En iyi bireyi güncelle
        if fitness(1) < best_fitness
            best_fitness = fitness(1);
            best_solution = population(1, :);
        end
        
        convergence_curve(iter) = best_fitness;
    end
end

% Benchmark fonksiyonları
function fitness = benchmark_func(population, func_num)
    fitness = zeros(size(population, 1), 1);
    
    switch func_num
        case 1
            % Sphere fonksiyonu
            fitness = sum(population.^2, 2);
        case 2
            % Rosenbrock fonksiyonu
            fitness = sum(100 * (population(:, 2:end) - population(:, 1:end-1).^2).^2 + (population(:, 1:end-1) - 1).^2, 2);
        % Diğer benchmark fonksiyonları buraya eklenebilir
    end
end

% Parametreler
pop_size = 20;
max_iter = 1000;
dimensions = [10, 20, 30, 100, 500, 1000];
num_functions = 13;
num_runs = 30;

% Sonuçları saklamak için değişkenler
results = cell(length(dimensions), num_functions);
convergence = cell(length(dimensions), num_functions);
time_elapsed = zeros(length(dimensions), num_functions, num_runs);
stats = cell(length(dimensions), num_functions);

% Her boyut için döngü
for d = 1:length(dimensions)
    dim = dimensions(d);
    
    % Her fonksiyon için döngü
    for f = 1:num_functions
        
        % 30 bağımsız çalışma için döngü
        for r = 1:num_runs
            tic; % Zaman ölçümü başlat
            
            % Aritmetik Optimizasyon Algoritması'nı çalıştır
            [best_fitness, convergence_curve] = arithmetic_optimization(dim, pop_size, max_iter, f);
            
            time_elapsed(d, f, r) = toc; % Zaman ölçümü bitir
            
            % Sonuçları sakla
            results{d, f}(r) = best_fitness;
            convergence{d, f}(r, :) = convergence_curve;
        end
        
        % İstatistiksel hesaplamalar
        stats{d, f}.mean = mean(results{d, f});
        stats{d, f}.std = std(results{d, f});
        stats{d, f}.median = median(results{d, f});
        stats{d, f}.best = min(results{d, f});
        stats{d, f}.worst = max(results{d, f});
        
        % Grafik çizimleri
        figure;
        plot(mean(convergence{d, f}));
        title(sprintf('Convergence Plot - Function %d, Dimension %d', f, dim));
        xlabel('Iteration');
        ylabel('Fitness');
        
        figure;
        boxplot(results{d, f});
        title(sprintf('Box Plot - Function %d, Dimension %d', f, dim));
        ylabel('Fitness');
    end
end

% Wilcoxon İşaretli Sıralama Testi
p_values = zeros(length(dimensions), num_functions);
for d = 1:length(dimensions)
    for f = 1:num_functions
        [~, p_values(d, f)] = signrank(results{d, f});
    end
end

% Sonuçları tablolar halinde yazdır
for d = 1:length(dimensions)
    fprintf('Dimension: %d\n', dimensions(d));
    fprintf('Function\tMean\tStd\tMedian\tBest\tWorst\tp-value\n');
    for f = 1:num_functions
        fprintf('%d\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e\n', f, stats{d, f}.mean, stats{d, f}.std, stats{d, f}.median, stats{d, f}.best, stats{d, f}.worst, p_values(d, f));
    end
    fprintf('\n');
end
