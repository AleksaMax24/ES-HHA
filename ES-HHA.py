import numpy as np
import math

class ES_HHA:
    def __init__(self, population_size=100, dimensions=30, max_FEs=30000, w1=0.5, R=1.0, F=0.8):
        # Инициализация параметров HH
        self.population_size = population_size
        self.dimensions = dimensions
        self.max_FEs = max_FEs
        self.w1 = w1
        self.R = R
        self.F = F
        
        # Инициализация LLH пулов
        self.exploitation_pool = self.initialize_exploitation_pool()
        self.exploration_pool = self.initialize_exploration_pool()
        
    def initialize_exploitation_pool(self):
        # LLH1-LLH4 - операторы эксплуатации
        return ['uniform', 'normal', 'levy', 'DE_best_1']
    
    def initialize_exploration_pool(self):
        # LLH5-LLH8 - операторы исследования  
        return ['DE_rand_1', 'DE_cur_1', 'DE_cur_to_best_1', 'DE_cur_to_pbest_1']
    
    def initialize_population(self, lb, ub):
        # Инициализация популяции
        population = np.random.uniform(lb, ub, (self.population_size, self.dimensions))
        fitness = np.array([self.objective_function(ind) for ind in population])
        best_idx = np.argmin(fitness)
        return population, fitness, population[best_idx], fitness[best_idx]
    
    def calculate_FDC(self, population, fitness, best_solution):
        # calculate_FDC
        distances = np.array([np.sqrt(np.sum((ind - best_solution)**2)) for ind in population])
        f_mean, d_mean = np.mean(fitness), np.mean(distances)
        numerator = np.sum((fitness - f_mean) * (distances - d_mean))
        denominator = np.sqrt(np.sum((fitness - f_mean)**2) * np.sum((distances - d_mean)**2))
        return numerator / denominator if denominator != 0 else 0
    
    def calculate_PD(self, population, lb, ub):
        # calculate_PD
        centroid = np.mean(population, axis=0)
        diversity_sum = np.sum([abs(population[i,j] - centroid[j]) / (ub[j] - lb[j]) 
                               for i in range(len(population)) for j in range(self.dimensions)])
        return diversity_sum / (len(population) * self.dimensions)
    
    def calculate_P1(self, FDC, PD):
        # calculate_P1
        P1 = self.w1 * FDC + (1 - self.w1) * PD
        return max(0.1, min(0.9, P1))
    
    def select_LLH(self, P1):
        # Выбор типа пула
        if np.random.random() < P1:
            pool = self.exploitation_pool
        else:
            pool = self.exploration_pool
        return np.random.choice(pool)
    
    def apply_LLH(self, LLH_name, population, best_solution, current_index):
        # apply_LLH
        current_ind = population[current_index]
        n = len(population)
        
        if LLH_name == 'uniform':
            return best_solution + self.R * np.random.uniform(-1, 1, self.dimensions)
        elif LLH_name == 'normal':
            return best_solution + self.R * np.random.normal(0, 1, self.dimensions)
        elif LLH_name == 'levy':
            # Реализация Levy flight
            return best_solution  # заглушка
        elif 'DE' in LLH_name:
            # DE операторы
            return current_ind  # заглушка
    
    def greedy_selection(self, current_ind, new_ind, current_fitness, new_fitness):
        # greedy_selection
        if new_fitness < current_fitness:
            return new_ind, new_fitness, True
        return current_ind, current_fitness, False
    
    def check_stopping_criteria(self, current_FEs, iteration, stagnation_count):
        # check_stopping_criteria
        if current_FEs >= self.max_FEs:
            return True, "Max FEs"
        return False, "Continue"
    
    def objective_function(self, individual):
        # Оценка приспособленности
        return np.sum(individual**2)  # сфера для примера
    
    def optimize(self, lb, ub):
        # Основной цикл ES-HHA
        
        # Инициализация популяции
        population, fitness, best_solution, best_fitness = self.initialize_population(lb, ub)
        current_FEs = self.population_size
        iteration = 0
        
        while True:
            # calculate_FDC + calculate_PD + calculate_P1
            FDC = self.calculate_FDC(population, fitness, best_solution)
            PD = self.calculate_PD(population, lb, ub)
            P1 = self.calculate_P1(FDC, PD)
            
            for i in range(self.population_size):
                # Выбор типа пула + apply_LLH
                LLH_name = self.select_LLH(P1)
                new_individual = self.apply_LLH(LLH_name, population, best_solution, i)
                new_individual = np.clip(new_individual, lb, ub)
                
                # Оценка приспособленности
                new_fitness = self.objective_function(new_individual)
                current_FEs += 1
                
                # greedy_selection
                population[i], fitness[i], improved = self.greedy_selection(
                    population[i], new_individual, fitness[i], new_fitness)
                
                # Обновление best_solution
                if improved and fitness[i] < best_fitness:
                    best_solution, best_fitness = population[i].copy(), fitness[i]
            
            # check_stopping_criteria
            stop, reason = self.check_stopping_criteria(current_FEs, iteration, 0)
            if stop:
                break
                
            iteration += 1
        
        # Вывод результатов
        return {
            'best_solution': best_solution,
            'best_fitness': best_fitness,
            'total_FEs': current_FEs,
            'iterations': iteration
        }

if __name__ == "__main__":
    es_hha = ES_HHA()
    lb = np.full(30, -100)
    ub = np.full(30, 100)
    results = es_hha.optimize(lb, ub)
    print(f"Best fitness: {results['best_fitness']}")