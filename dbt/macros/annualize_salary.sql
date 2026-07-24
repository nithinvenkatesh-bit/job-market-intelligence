{#
    Convert a pay figure to an annual equivalent.

    This lives in a macro rather than being repeated inline because the
    conversion appears in several models, and the hours-per-year assumption
    is exactly the kind of thing that silently diverges when copy-pasted.
    One definition, one place to change it.

    HOURS_PER_YEAR = 2080 is 40h x 52w. Part-time roles are overstated by
    this, which is a documented limitation rather than a hidden one.
#}

{% macro annualize_salary(amount_column, period_column) %}
    case
        when {{ amount_column }} is null then null
        when {{ period_column }} = 'HOURLY'   then {{ amount_column }} * 2080
        when {{ period_column }} = 'WEEKLY'   then {{ amount_column }} * 52
        when {{ period_column }} = 'BIWEEKLY' then {{ amount_column }} * 26
        when {{ period_column }} = 'MONTHLY'  then {{ amount_column }} * 12
        when {{ period_column }} = 'YEARLY'   then {{ amount_column }}
        else null
    end
{% endmacro %}