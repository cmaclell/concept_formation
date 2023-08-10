library(readr) # Read the csv file
library(ggplot2)  # Basic plot package



exp0 <- read_csv("Documents/GitHub/catastrophic-forgetting-cobweb/experiments/r_plots/exp0.csv",
                    col_types = cols(Seed = col_character()))  # read the file and import it to a dataframe
# Change the name of '0-9':
exp0$TestSet[exp0$TestSet == '0-9'] <- 'all'
# Change the name of x-axis (Trainset):
exp0$TrainSet[exp0$TrainSet == '0-9: 1'] <- 1
exp0$TrainSet[exp0$TrainSet == '0-9: 2'] <- 2
exp0$TrainSet[exp0$TrainSet == '0-9: 3'] <- 3
exp0$TrainSet[exp0$TrainSet == '0-9: 4'] <- 4
exp0$TrainSet[exp0$TrainSet == '0-9: 5'] <- 5
exp0$TrainSet[exp0$TrainSet == '0-9: 6'] <- 6
exp0$TrainSet[exp0$TrainSet == '0-9: 7'] <- 7
exp0$TrainSet[exp0$TrainSet == '0-9: 8'] <- 8
exp0$TrainSet[exp0$TrainSet == '0-9: 9'] <- 9
exp0$TrainSet[exp0$TrainSet == '0-9: 10'] <- 10
# Convert TrainSet to factor and specify the desired order of levels
exp0$TrainSet <- factor(exp0$TrainSet, levels = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10))  # Adjust the levels as needed
exp0$Model[exp0$Model == "COBWEB/4T"] <- "COBWEB/4V"

df_0 <- exp0 %>%
  filter(TestSet %in% c('0', 'all'))  # Only with the specified label (0) and 'all'

p <- ggplot(exp0, aes(x = TrainSet, y = TestAccuracy, color = TestSet, fill = TestSet, group = TestSet)) +
  #geom_line() +
  #geom_ribbon(aes(ymin = TestAccuracy - sd(TestAccuracy), ymax = TestAccuracy + sd(TestAccuracy)), alpha = 0.2) +
  #geom_ribbon(aes(ymin = quantile(TestAccuracy, 0.025), ymax = quantile(TestAccuracy, 0.975)), alpha = 0.2) +
  stat_summary(
    fun.data = function(y) data.frame(y = mean(y), ymin = quantile(y, 0.025), ymax = quantile(y, 0.975)),
    geom = "ribbon",
    method = "lm",
    formula = y ~ x,
    aes(group = TestSet),
    se = FALSE,
    linetype = "blank",
    size = 1,
    alpha = 0.5
  ) +  # Adding bootstrap confidence intervals
  stat_summary(fun.data = function(y) data.frame(y = median(y)),
               geom = "line",
               size = 1,
               alpha = 0.5) +
  labs(x = '# of Training Sets', y = 'Test Accuracy', title = 'Test Accuracy with Increasing Training Data from all Labels') +
  theme_minimal() +
  facet_wrap(~ Model, ncol = 3) +
  scale_color_manual(values = c('0' = 'red', '1' = 'orange', '2' = 'green', '3' = 'blue', '4' = 'pink',
                                '5' = 'yellow', '6' = 'brown', '7' = 'purple', '8'='darkgreen', '9' = 'lightblue',
                                'all' = 'black')) +
  scale_fill_manual(values = c('0' = 'red', '1' = 'orange', '2' = 'green', '3' = 'blue', '4' = 'pink',
                               '5' = 'yellow', '6' = 'brown', '7' = 'purple', '8'='darkgreen', '9' = 'lightblue',
                               'all' = 'black')) +
  theme(plot.title = element_text(hjust = 0.5))  # Center the title
print(p)

# With error bars and lines only
p_err <- ggplot(exp0, aes(x = TrainSet, y = TestAccuracy, color = TestSet, fill = TestSet, group = TestSet)) +
  #geom_line() +
  #geom_ribbon(aes(ymin = TestAccuracy - sd(TestAccuracy), ymax = TestAccuracy + sd(TestAccuracy)), alpha = 0.2) +
  #geom_ribbon(aes(ymin = quantile(TestAccuracy, 0.025), ymax = quantile(TestAccuracy, 0.975)), alpha = 0.2) +
  geom_errorbar(stat = "summary",
                fun.data = function(y) data.frame(ymin = median(y) - sd(y), ymax = median(y) + sd(y)),
                width = 0.2) +
  stat_summary(fun.data = function(y) data.frame(y = median(y)),
               geom = "line",
               size = 1,
               alpha = 0.5) +
  labs(x = '# of Training Sets', y = 'Test Accuracy', title = 'Test Accuracy with Increasing Training Data from all Labels') +
  theme_minimal() +
  facet_wrap(~ Model, ncol = 3) +
  scale_color_manual(values = c('0' = 'red', '1' = 'orange', '2' = 'green', '3' = 'blue', '4' = 'pink',
                                '5' = 'yellow', '6' = 'brown', '7' = 'purple', '8'='darkgreen', '9' = 'lightblue',
                                'all' = 'black')) +
  scale_fill_manual(values = c('0' = 'red', '1' = 'orange', '2' = 'green', '3' = 'blue', '4' = 'pink',
                               '5' = 'yellow', '6' = 'brown', '7' = 'purple', '8'='darkgreen', '9' = 'lightblue',
                               'all' = 'black')) +
  theme(plot.title = element_text(hjust = 0.5))  # Center the title
print(p_err)


p <- ggplot(df_0, aes(x = TrainSet, y = TestAccuracy, color = TestSet, fill = TestSet, group = TestSet)) +
  #geom_line() +
  #geom_ribbon(aes(ymin = TestAccuracy - sd(TestAccuracy), ymax = TestAccuracy + sd(TestAccuracy)), alpha = 0.2) +
  #geom_ribbon(aes(ymin = quantile(TestAccuracy, 0.025), ymax = quantile(TestAccuracy, 0.975)), alpha = 0.2) +
  geom_errorbar(stat = "summary",
                fun.data = function(y) data.frame(ymin = median(y) - sd(y), ymax = median(y) + sd(y)),
                width = 0.2) +
  stat_summary(
    fun.data = function(y) data.frame(y = mean(y), ymin = quantile(y, 0.025), ymax = quantile(y, 0.975)),
    geom = "ribbon",
    method = "lm",
    formula = y ~ x,
    aes(group = TestSet),
    se = FALSE,
    linetype = "blank",
    size = 1,
    alpha = 0.5
  ) +  # Adding bootstrap confidence intervals
  stat_summary(fun.data = function(y) data.frame(y = median(y)),
               geom = "line",
               size = 1,
               alpha = 0.5) +
  labs(x = '# of Training Sets', y = 'Test Accuracy', title = 'Test Accuracy with Increasing Training Data from all Labels') +
  theme_minimal() +
  facet_wrap(~ Model, ncol = 3) +
  scale_color_manual(values = c('0' = 'red', 'all' = 'blue')) +
  scale_fill_manual(values = c('0' = 'red', 'all' = 'blue')) +
  theme(plot.title = element_text(hjust = 0.5))  # Center the title
print(p)

df_all <- exp0 %>%
  filter(TestSet %in% c('all'))
p_all <- ggplot(df_all, aes(x = TrainSet, y = TestAccuracy, color = Model, fill = Model, group = Model)) +
  #geom_line() +
  #geom_ribbon(aes(ymin = TestAccuracy - sd(TestAccuracy), ymax = TestAccuracy + sd(TestAccuracy)), alpha = 0.2) +
  #geom_ribbon(aes(ymin = quantile(TestAccuracy, 0.025), ymax = quantile(TestAccuracy, 0.975)), alpha = 0.2) +
  geom_errorbar(stat = "summary",
                #fun.data = function(y) data.frame(ymin = median(y) - sd(y), ymax = median(y) + sd(y)),
                fun.data = "mean_cl_boot",
                width = 0.2) +
  stat_summary(#fun.data = function(y) data.frame(y = median(y)),
               fun.data = "mean_cl_boot",
               geom = "line",
               size = 1,
               alpha = 0.5) +
  labs(x = '# of Training Sets', y = 'Test Accuracy', title = 'Test Accuracy of all Labels with Increasing Training Data from all Labels') +
  theme_minimal() +
  #facet_wrap(~ Model, ncol = 3) +
  scale_color_manual(values = c('COBWEB/4V' = 'red', 'fc' = 'darkgreen', 'fc-CNN' = 'blue')) +
  scale_fill_manual(values = c('COBWEB/4V' = 'red', 'fc' = 'darkgreen', 'fc-CNN' = 'blue')) +
  theme(plot.title = element_text(hjust = 0.5))  # Center the title
print(p_all)
