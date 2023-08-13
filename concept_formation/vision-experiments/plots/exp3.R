library(readr)
library(readr) # Read the csv file
library(ggplot2)  # Basic plot package

exp3 <- read_csv("Documents/GitHub/catastrophic-forgetting-cobweb/experiments/r_plots/Data/exp3.csv")

exp3$TrainSet <- factor(exp3$TrainSet, levels = c('D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10'))
exp3$TestSet <- factor(exp3$TestSet, levels = c('L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'All'))

# Calculate the averages among "rest labels"
df_avg_3 <- exp3 %>% 
  filter(TestSet %in% c("L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9")) %>%
  group_by(TrainSet, Model, Seed) %>% 
  summarize(AvgTestAccuracy = mean(TestAccuracy))

# Then remove the "rest labels"
df_filtered_3 <- exp3 %>%
  filter(!TestSet %in% c("L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9"))

df_rest_3 <- df_avg_3 %>%
  mutate(TestSet = "Rest", TestAccuracy = AvgTestAccuracy) %>%
  select(-AvgTestAccuracy)

df_3 <- rbind(df_rest_3, df_filtered_3[,c("TrainSet", "Model", "Seed", "TestSet", "TestAccuracy")])
df_3$TestSet[df_3$TestSet == "L0"] <- "Chosen"
df_3$TestSet <- factor(df_3$TestSet, levels = c('Chosen', 'Rest', 'All'))

df_chosen_3 = df_3 %>% 
  filter(TestSet == 'Chosen')

p_chosen_3 <- ggplot(df_chosen_3, aes(x = TrainSet, y = TestAccuracy, color = Model, fill = Model, group = Model)) +
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
  labs(x = 'Incoming Training Split', y = 'Test Accuracy', 
       title = 'Experiment 3') +
  theme_minimal() +
  #facet_wrap(~ Model, ncol = 3) +
  scale_color_manual(values = c('cobweb4v' = 'red', 'fc' = 'darkgreen', 'fc-cnn' = 'blue')) +
  scale_fill_manual(values = c('cobweb4v' = 'red', 'fc' = 'darkgreen', 'fc-cnn' = 'blue')) +
  theme(plot.title = element_text(hjust = 0.5))  # Center the title
print(p_chosen_3)

df_display_3 <- ggplot_build(p_chosen_3)$data[[2]]
