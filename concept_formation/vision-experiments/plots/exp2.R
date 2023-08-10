library(readr) # Read the csv file
library(ggplot2)  # Basic plot package
library(dplyr)

exp2 <- read_csv("Documents/GitHub/catastrophic-forgetting-cobweb/experiments/r_plots/Data/exp2.csv")

exp2$TrainSet <- factor(exp2$TrainSet, levels = c('S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9'))
exp2$TestSet <- factor(exp2$TestSet, levels = c('L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'All'))

# Calculate the averages among "rest labels"
df_avg <- exp2 %>% 
  filter(TestSet %in% c("L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9")) %>%
  group_by(TrainSet, Model, Label, Seed) %>% 
  summarize(AvgTestAccuracy = mean(TestAccuracy))

# Then remove the "rest labels"
df_filtered <- exp2 %>%
  filter(!TestSet %in% c("L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9"))

df_rest <- df_avg %>%
  mutate(TestSet = "Rest", TestAccuracy = AvgTestAccuracy) %>%
  select(-AvgTestAccuracy)

df <- rbind(df_rest, df_filtered[,c("TrainSet", "Model", "Label", "Seed", "TestSet", "TestAccuracy")])
df$TestSet[df$TestSet == "L0"] <- "Chosen"
df$TestSet <- factor(df$TestSet, levels = c('Chosen', 'Rest', 'All'))

df_chosen = df %>% 
  filter(TestSet == 'Chosen')

p_chosen <- ggplot(df_chosen, aes(x = TrainSet, y = TestAccuracy, color = Model, fill = Model, group = Model)) +
  #geom_line() +
  #geom_ribbon(aes(ymin = TestAccuracy - sd(TestAccuracy), ymax = TestAccuracy + sd(TestAccuracy)), alpha = 0.2) +
  #geom_ribbon(aes(ymin = quantile(TestAccuracy, 0.025), ymax = quantile(TestAccuracy, 0.975)), alpha = 0.2) +
  #geom_errorbar(stat = "summary",
  #fun.data = function(y) data.frame(ymin = median(y) - sd(y), ymax = median(y) + sd(y)),
  #width = 0.2) +
  geom_errorbar(stat = "summary",
                #fun.data = function(y) data.frame(ymin = median(y) - sd(y), ymax = median(y) + sd(y)),
                fun.data = "mean_cl_boot",
                width = 0.2) +
  stat_summary(#fun.data = function(y) data.frame(y = median(y)),
    fun.data = "mean_cl_boot",
    geom = "line",
    size = 1,
    alpha = 0.5) +
  labs(x = 'Incoming Training Split', y = 'Test Accuracy', title = 'Test Accuracy for Chosen Label with More Pre-defined Training Splits, Experiment 2') +
  theme_minimal() +
  #facet_wrap(~ Model, ncol = 3) +
  scale_color_manual(values = c('COBWEB/4V' = 'red', 'fc' = 'darkgreen', 'fc-CNN' = 'blue')) +
  scale_fill_manual(values = c('COBWEB/4V' = 'red', 'fc' = 'darkgreen', 'fc-CNN' = 'blue')) +
  theme(plot.title = element_text(hjust = 0.5))  # Center the title
print(p_chosen)

