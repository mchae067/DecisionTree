import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Stack;

/**
 * Fill in the implementation details of the class DecisionTree using this file.
 * Any methods or secondary classes that you want are fine but we will only
 * interact with those methods in the DecisionTree framework.
 * 
 * You must add code for the 5 methods specified below.
 * 
 * See DecisionTree for a description of default methods.
 */
public class DecisionTreeImpl extends DecisionTree {
	private DecTreeNode root;
	private List<String> labels; // ordered list of class labels
	private List<String> attributes; // ordered list of attributes
	private Map<String, List<String>> attributeValues; // map to ordered
														// discrete values taken
														// by attributes

	/**
	 * Answers static questions about decision trees.
	 */
	DecisionTreeImpl() {
		// no code necessary
		// this is void purposefully
	}

	/**
	 * Build a decision tree given only a training set.
	 * 
	 * @param train: the training set
	 */
	DecisionTreeImpl(DataSet train) {
		this.labels=train.labels;
		this.attributes=train.attributes;
		this.attributeValues=train.attributeValues;
		// TODO: add code here
        root=buildTree(train.instances, copyAttributes(attributes), 1, 1);
	}
	
	//1.2.2
    private DecTreeNode buildTree (List<Instance> instances, List<String> attrs,
    		Integer defaultLabel, Integer value) {
        if (instances.isEmpty()) {
            return new DecTreeNode(defaultLabel, null, value, true);
        }
        if (attrs.isEmpty()) {
        	return new DecTreeNode(getPlurality(instances), null, value, true);
        }
        Integer label=instances.get(0).label;
        boolean sameLabel=true;
        for (Instance instance:instances) {
            if (instance.label.intValue()!=label.intValue()) {
                sameLabel=false;
            }
        }
        if (sameLabel) {
            return new DecTreeNode(label, null, value, true);
        }
        float maxGain=-1;
        String maxGainAttribute="";
        for (String attr:attrs) {
            float gain=gainInfo(attr, instances);
            if (gain>maxGain) {
                maxGainAttribute=attr;
                maxGain=gain;
            }
        }
        int AttrIndex=0, index=0;
        AttrIndex=attributes.indexOf(maxGainAttribute);
        DecTreeNode current;
        label=getPlurality(instances);
        current=new DecTreeNode(new Integer(label), new Integer(AttrIndex),
        		value, false);
        List<String> AttrVals=attributeValues.get(maxGainAttribute);
        if (AttrVals==null) {
            return current;
        }
        for (String val:AttrVals) {
            index=AttrVals.indexOf(val);
            List<Instance> filteredInst=new ArrayList<Instance>();
            for (Instance instance:instances) {
                if (instance.attributes.get(AttrIndex).intValue()==index) {
                    filteredInst.add(instance);
                }
            }
            List<String> filteredAttrs=copyAttributes(attrs);
            filteredAttrs.remove(maxGainAttribute);
            current.children.add(buildTree(
            		filteredInst, filteredAttrs, defaultLabel, index));
        }

        return current;
    }
    
    //Doesn't work if attributes is used alone...
    private List<String> copyAttributes(List<String> attributes) {
        List<String> attrs=new ArrayList<String>();
        for (String attr:attributes) {
            attrs.add(attr);
        }
        return attrs;
    }

    
	/**
	 * Build a decision tree given a training set then prune it using a tuning
	 * set.
	 * 
	 * @param train: the training set
	 * @param tune: the tuning set
	 */
	DecisionTreeImpl(DataSet train, DataSet tune) {
		// TODO: add code here
        this(train);
        tune(tune);
	}
	
	//1.2.3
    private void tune(DataSet tune) {
        Stack<DecTreeNode> stack=new Stack<DecTreeNode>();
        stack.push(root);
        float max=0.0f;
        DecTreeNode bestNode=null;
        while (!stack.isEmpty()) {
            DecTreeNode node=stack.pop();
            if (node.children!=null) {
                for (DecTreeNode child:node.children) {
                    stack.push(child);
                }
            }
            if (!node.terminal) {
                node.terminal=true;
                float accuracy=accuracy(tune);
                if (accuracy>max) {
                    max=accuracy;
                    bestNode=node;
                }
                node.terminal=false;
            }
        }
        bestNode.terminal=true;

        return;
    }


    //1.3
	@Override
	public String classify(Instance instance) {
		// TODO: add code here
        DecTreeNode node=root;
        boolean hasChanged=true;
        while (!node.terminal && hasChanged) {
            hasChanged=false;
            for (DecTreeNode child:node.children) {
                if (child.parentAttributeValue.intValue()==instance.attributes
                		.get(node.attribute.intValue()).intValue()) {
                    hasChanged=true;
                    node=child;
                    break;
                }
            }
        }
		return labels.get(node.label);
	}

	
	@Override
	/**
	 * Print the decision tree in the specified format
	 */
	public void print() {
		printTreeNode(root, null, 0);
	}
	
	
	/**
	 * Prints the subtree of the node
	 * with each line prefixed by 4 * k spaces.
	 */
	public void printTreeNode(DecTreeNode p, DecTreeNode parent, int k) {
		StringBuilder sb=new StringBuilder();
		for (int i=0; i<k; i++) {
			sb.append("    ");
		}
		String value;
		if (parent==null) {
			value="ROOT";
		} else{
			String parentAttribute=attributes.get(parent.attribute);
			value=attributeValues.get(parentAttribute)
					.get(p.parentAttributeValue);
		}
		sb.append(value);
		if (p.terminal) {
			sb.append(" (" + labels.get(p.label) + ")");
			System.out.println(sb.toString());
		} else {
			sb.append(" {" + attributes.get(p.attribute) + "?}");
			System.out.println(sb.toString());
			for(DecTreeNode child: p.children) {
				printTreeNode(child, p, k+1);
			}
		}
	}

	//1.3.2
	@Override
	public void rootInfoGain(DataSet train) {		
		this.labels=train.labels;
		this.attributes=train.attributes;
		this.attributeValues=train.attributeValues;
		// TODO: add code here
        for (String attribute:attributes) {
            System.out.format("%s %.5f\n", attribute, 
            		gainInfo(attribute, train.instances));
        }
	}

    private float gainInfo(String attr, List<Instance> instances) {
        return getEntropy(instances) - getEntropy(instances, attr);
    }

	
    private float getEntropy(List<Instance> instances) {
        float sum=0.0f;
        for (String label:labels) {
            int matches=0;
            int labelIndex=labels.indexOf(label);
            for (Instance instance:instances) {
                if (instance.label.intValue()==labelIndex) {
                    ++matches;
                }
            }
            float prob=((float)matches)/((float)instances.size());
            if (instances.size()!=0 && prob>0.0f) {
                sum -= prob * Math.log(prob)/Math.log(2.0);
            }

        }
        return sum;
    }


    private float getEntropy(List<Instance> instances, String attr) {
        float sum=0.0f;
        for (String value:attributeValues.get(attr)) {
            int matches=0;
            for (Instance instance:instances) {
                if (instance.attributes.get(attributes.indexOf(attr))
                		.intValue()==attributeValues.get(attr).indexOf(value)) {
                    ++matches;
                }
            }
            if (instances.size()!=0) {
                sum += ((float)matches)/((float)instances.size())*
                		getEntropy(instances, attr, value);
            }
        }
        return sum;
    }


    private float getEntropy(List<Instance> instances, 
    		String attr, String value) {
        float sum=0.0f;
        List<Instance> matching=new ArrayList<Instance>();
        for (Instance instance:instances) {
            if (instance.attributes.get(attributes.indexOf(attr)).
            		intValue()==attributeValues.get(attr).indexOf(value)) {
                matching.add(instance);
            }
        }
        for (String label:labels) {
            int matches=0;
            int labelIndex=labels.indexOf(label);
            for (Instance instance:matching) {
                if (instance.label.intValue()==labelIndex) {
                    ++matches;
                }
            }
            float prob=((float)matches)/((float)matching.size());
            if (matching.size()!=0 && prob>0.0f) {
                sum -= prob * Math.log(prob)/Math.log(2.0);
            }
        }
        return sum;
    }
    
   
    private float accuracy(DataSet tune) {
        int hit=0;
        for (Instance instance:tune.instances) {
            int labelIndex=instance.label.intValue();
            String instanceLabel=tune.labels.get(labelIndex);
            if (instanceLabel.equals(classify(instance))) {
                ++hit;
            }
        }
        return ((float)hit)/((float)tune.instances.size());
    }

    
    private int getPlurality(List<Instance> instances) {
        ArrayList<Integer> labels=new ArrayList<Integer>();
        ArrayList<Integer> count=new ArrayList<Integer>();
        for (Instance instance: instances) {
            if (labels.contains(instance.label)) {
                int labelIndex=labels.indexOf(instance.label);
                count.set(labelIndex, count.get(labelIndex) + 1);
            } else {
                labels.add(instance.label);
                count.add(1);
            }
        }
        Integer max=-1;
        Integer pluralityLabel=-1;
        for (int i=0; i<count.size(); i++) {
            if (max<count.get(i)) {
                max=count.get(i);
                pluralityLabel=labels.get(i);
            }
        }
        return pluralityLabel;
    }
}