Feature: Primitive Combinator
  The primitive combinator wraps a program and performs traced evaluation, as
  well as traced evaluation under substitution.

  Scenario: Construction
    Given a pyro model
    When it has no observe statements

    Then I get a primitive inference combinator

#  Scenario: Traced evaluation
#    Given a primitive inference combinator
#    When I evaluate it with the appropriate arguments
#
#    Then I get the original model outputs
#    And a trace (which includes the a density map and sampled values)
#    And a likelihood weight of 1
#
#  Scenario: Traced evaluation under substitution with no overlaps
#    Given a primitive inference combinator
#    And a second primitive inference combinator
#    And the second primitive has no overlapping addresses
#    And the second primitive is used as a substitution context
#
#    Then I get the original model outputs
#    And a trace (which includes the a density map and sampled values)
#    And a likelihood weight of 1
#
#  Scenario: Traced evaluation under substitution with overlaps
#    Given a primitive inference combinator
#    And a second primitive inference combinator
#    And the second primitive with an overlapping address
#    And the second primitive is used as a substitution context
#
#    Then I get the original model outputs
#    And a trace (which includes the a density map and sampled values)
#    And any overlapping addresses have values from the second primitive, not the first
#    And the output likelihood weight is not equal to 1
#
