%%%%%%%%%%%%%%%% experiment file  (contains experiment, receives genom, decodes it, evaluates it and returns raw fitnesses) (function)

%% Neuro_Evolution_of_Augmenting_Topologies - NEAT 
%% developed by Kenneth Stanley (kstanley@cs.utexas.edu) & Risto Miikkulainen (risto@cs.utexas.edu)
%% Coding by Christian Mayr (matlab_neat@web.de)
%% ADJUSTED FOR GENERALIZATION BY ETHAN PETERS (15esp3@queensu.ca)

function population_plus_fitnesses=experiment(population,genre);
population_plus_fitnesses=population;
no_change_threshold=1e-3; %threshold to judge if state of a node has changed significantly since last iteration
number_individuals=size(population,2);

dataset=csvread("datasets/"+genre+".csv");
[m,n] = size(dataset);
input_pattern = dataset(:,1:(n-1));

[total_patterns,input_nodes]=size(input_pattern);
           
output_pattern = dataset(:,n);

pop_error = zeros(1,number_individuals);

for index_individual=1:number_individuals
%    display("Individual: "+index_individual);
   
   number_nodes=size(population(index_individual).nodegenes,2);
   number_connections=size(population(index_individual).connectiongenes,2);
   individual_fitness=0;
   output=[];
   hidden_nodes = (input_nodes+2):number_nodes;
   
   for index_pattern=1:total_patterns
      
      
      % the following code assumes node 1 to 13 inputs, node 14 bias, node 15 output, rest arbitrary (if existent, will be hidden nodes)
      % set node input steps for first timestep
      
      population(index_individual).nodegenes(3,hidden_nodes)=0; %set all node input states to zero
      population(index_individual).nodegenes(3,input_nodes+1)=1; %bias node input state set to 1
      population(index_individual).nodegenes(3,1:input_nodes)=input_pattern(index_pattern,:); %node input states of the two input nodes are consecutively set to the given dataset input pattern  
      
      %set node output states for first timestep (depending on input states)
      population(index_individual).nodegenes(4,1:(input_nodes+1))=population(index_individual).nodegenes(3,1:(input_nodes+1));
      population(index_individual).nodegenes(4,hidden_nodes)=1./(1+exp(-4.9*population(index_individual).nodegenes(3,hidden_nodes)));
      no_change_count=0;     
      index_loop=0;
      while (no_change_count<number_nodes) & index_loop<3*number_connections
         index_loop=index_loop+1;
         vector_node_state=population(index_individual).nodegenes(4,:);
         for index_connections=1:number_connections
            %read relevant contents of connection gene (ID of Node where connection starts, ID of Node where it ends, and connection weight)
            ID_connection_from_node=population(index_individual).connectiongenes(2,index_connections);
            ID_connection_to_node=population(index_individual).connectiongenes(3,index_connections);
            connection_weight=population(index_individual).connectiongenes(4,index_connections);
            %map node ID's (as extracted from single connection genes above) to index of corresponding node in node genes matrix
            index_connection_from_node=find((population(index_individual).nodegenes(1,:)==ID_connection_from_node));
            index_connection_to_node=find((population(index_individual).nodegenes(1,:)==ID_connection_to_node));
                        
            if population(index_individual).connectiongenes(5,index_connections)==1 %Check if Connection is enabled
               population(index_individual).nodegenes(3,index_connection_to_node)=population(index_individual).nodegenes(3,index_connection_to_node)+population(index_individual).nodegenes(4,index_connection_from_node)*connection_weight; %take output state of connection_from node, multiply with weight, add to input state of connection_to node
            end
         end
         %pass on node input states to outputs for next timestep 
         population(index_individual).nodegenes(4,hidden_nodes)=1./(1+exp(-4.9*population(index_individual).nodegenes(3,hidden_nodes)));          
         %Re-initialize node input states for next timestep
         population(index_individual).nodegenes(3,hidden_nodes)=0; %set all output and hidden node input states to zero
         no_change_count=sum(abs(population(index_individual).nodegenes(4,:)-vector_node_state)<no_change_threshold); %check for alle nodes where the node output state has changed by less than no_change_threshold since last iteration through all the connection genes
      end      
%       if index_loop>=2.7*number_connections
%          index_individual
%          population(index_individual).connectiongenes
%       end
      output=[output;population(index_individual).nodegenes(4,(input_nodes+2))];
      individual_fitness=individual_fitness+abs(output_pattern(index_pattern,1)-population(index_individual).nodegenes(4,(input_nodes+2))); %prevent oscillatory connections from achieving high fitness   
      
%       display([output_pattern(index_pattern,1),population(index_individual).nodegenes(4,15)]);
      
   end
   
%    display("Fitness: "+(individual_fitness))
   population_plus_fitnesses(index_individual).fitness=(m-individual_fitness)^2; %Fitness function as defined by Kenneth Stanley, CONFUSED IF 15 IS THE CORRECT NUMBER
   
   displace = round(output)-output_pattern;
   
   if sum(abs(round(output)-output_pattern))==0
      display("PERFECT");
      population_plus_fitnesses(index_individual).fitness=m^2;
%       output;
   end
   
   
   pop_error(index_individual) = individual_fitness/total_patterns;
end
fileID = fopen("out_"+genre+".txt",'a');
fprintf(fileID,'%6.2f,',mean(pop_error));
fclose(fileID);

