library(readr)
library(readr) # Read the csv file
library(ggplot2)  # Basic plot package

exp4 <- read_csv("Documents/GitHub/catastrophic-forgetting-cobweb/experiments/r_plots/Data/exp4.csv")

exp4$TrainSet <- factor(exp4$TrainSet, levels = c('D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10'))
exp4$TestSet <- factor(exp4$TestSet, levels = c('L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'All'))

# Calculate the averages among "rest labels"
df_avg_4 <- exp4 %>% 
  filter(TestSet %in% c("L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9")) %>%
  group_by(TrainSet, Model, Seed) %>% 
  summarize(AvgTestAccuracy = mean(TestAccuracy))

# Then remove the "rest labels"
df_filtered_4 <- exp4 %>%
  filter(!TestSet %in% c("L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9"))

df_rest_4 <- df_avg_4 %>%
  mutate(TestSet = "Rest", TestAccuracy = AvgTestAccuracy) %>%
  select(-AvgTestAccuracy)

df_4 <- rbind(df_rest_4, df_filtered_4[,c("TrainSet", "Model", "Seed", "TestSet", "TestAccuracy")])
df_4$TestSet[df_4$TestSet == "L0"] <- "Chosen"
df_4$TestSet <- factor(df_4$TestSet, levels = c('Chosen', 'Rest', 'All'))

df_chosen_4 = df_4 %>% 
  filter(TestSet == 'Chosen')

p_chosen_4 <- ggplot(df_chosen_4, aes(x = TrainSet, y = TestAccuracy, color = Model, fill = Model, group = Model)) +
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
       title = 'Experiment 4') +

  theme_minimal() +
  #facet_wrap(~ Model, ncol = 3) +
  scale_color_manual(values = c('cobweb4v' = 'red', 'fc' = 'darkgreen', 'fc-cnn' = 'blue')) +
  scale_fill_manual(values = c('cobweb4v' = 'red', 'fc' = 'darkgreen', 'fc-cnn' = 'blue')) +
  theme(plot.title = element_text(hjust = 0.5))  # Center the title
print(p_chosen_4)

ggplot_build(p_chosen_4)$data
df_display_all <- ggplot_build(p_chosen_4)$data[[2]]  # data for stat_summary line

df_display_4 <- ggplot_build(p_chosen_4)$data[[2]]

