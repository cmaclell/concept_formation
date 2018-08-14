var ClusterRenderer = /** @class */ (function() {

    var w = 750;
    var h = 750;
    var padding = 50;
    var xScale = d3.scaleLinear().range([padding*1.5,w-padding*1.5]).domain([0,1]);
    var yScale = d3.scaleLinear().range([h-padding*1.5,padding*1.5]).domain([0,1]);
    var sizeScale = d3.scaleLinear().range([padding/5,padding*1.5]);
    var local_data;
    var local_clusters;
    var distance_matrix;
    var nodes;


    var vis = d3.select("svg");
    var xAxis = d3.axisBottom().scale(xScale);
    var yAxis = d3.axisLeft().scale(yScale);

    var av_properties = {};
    var saturationScale = d3.scaleLinear()
        .domain([0, 1])
        .range(["#cccccc", "steelblue"]);
    var defaultGrey = "#cccccc";

    function ClusterRenderer(rootObj) {
        local_clusters = window.extra_data.clusters;
        distance_matrix = window.extra_data.cluster_distance;
        var attrs = Object.keys(rootObj.counts);
        local_data = rootObj;

        av_properties["none"] = {
            "attrType": "constant",
            "valType": "default",
            "scale": null,
            "axisScale":d3.scaleLinear(xScale).domain([0,local_clusters.length+1])
        };
        for (var i = 0; i < attrs.length; i++) {
            var attr = attrs[i];
            var vals = Object.keys(rootObj.counts[attr]);

            var at_scale = {
                "attrType": vt.attrType(rootObj, attr),
                "valType": vt.valType(rootObj, attr),
                "scale": null,
                "axisScale": null,
                "filterTypes": vt.filterTypes(attr)
            };
            switch (at_scale.valType) {
                case "nominal":
                    at_scale.scale = d3.scaleOrdinal(d3.schemeCategory10).domain(vals);
                    at_scale.axisScale = d3.scaleOrdinal(xScale).domain(vals);
                    break;
                case "nominal20":
                    at_scale.scale = d3.scaleOrdinal(d3.schemeSet3).domain(vals);
                    at_scale.axisScale = d3.scaleOrdinal(xScale).domain(vals);
                    break;
                case "binary":
                    at_scale.scale = d3.scaleSequential(d3.interpolateViridis);
                    at_scale.axisScale = d3.scaleSequential(xScale).domain([0,1]);
                    at_scale.cat_scale = d3.scaleOrdinal(d3.schemeCategory10).domain(vals);
                    at_scale.num_scale = d3.scaleSequential(d3.interpolateViridis);
                    break;
                case "numeric":
                    var mm = vt.min_max(rootObj, attr, [Infinity, -Infinity]);
                    if (mm[0] === mm[1]) {
                        at_scale.valType = "constant-numeric";
                        at_scale.scale = saturationScale;
                        at_scale.axisScale = d3.scaleLinear(xScale).domain([mm[0]-.01,mm[1]+.01]);
                    } else {
                        at_scale.scale = d3.scaleSequential(d3.interpolateViridis).domain([mm[0], mm[1]]);
                        at_scale.axisScale = d3.scaleLinear(xScale).domain([mm[0], mm[1]]);
                    }
                    break;
                case "constant":
                    at_scale.scale = saturationScale;
                    at_scale.axisScale = xScale;
                    break;
                case "unique":
                    at_scale.axisScale = d3.scaleLinear(xScale).domain([0,local_clusters.length+1]);
                    break;
            }
            av_properties[attr] = at_scale;
        }


        local_clusters.sort();
        local_clusters.reverse();

        for (var i = 0; i < local_clusters.length; i ++) {
            local_clusters[i].index = i+1;
            local_clusters[i].perc = local_clusters[i].size / local_data.size;
        }


        setupClusterOptions(local_data,local_clusters);

    }

    function setupClusterOptions(tree_data,clusters){
        //wire up the UI stuff
        $('#dist-cluster-settings').hide();
        $('#cluster-style').change(function(e){
            if($('#cluster-style').val() === 'attr') {
                $('#attr-cluster-settings').show();
                $('#dist-cluster-settings').hide();
                updateAttrAxis();
                updateDistanceAxis();
            }
            else {
                $('#attr-cluster-settings').hide();
                $('#dist-cluster-settings').show();
                updateAttrAxis();
                updateDistanceAxis();
            }
        });

        $('.attr-axis').change(function(e){
            updateAttrAxis();
        });
        $('.dist-anchor').change(function(e) {
            updateDistanceAxis();
        });
        $('#cluster-sizing').change(function(e) {
            updateSizeChanged();
        });


        //populate attribute the axis options
        var attrs = Object.keys(tree_data.counts);
        var xSelect = $('#x-axis-attr');
        var ySelect = $('#y-axis-attr');

        xSelect.html('<option value="none">None</option>');
        ySelect.html('<option value="none">None</option>');

        for (var i = 0; i < attrs.length; i++) {
            var attrScale = av_properties[attrs[i]];
            var opt = $("<option>" + attrs[i] + "</option>");
            opt.attr("value", attrs[i]);
            if (attrScale) {
                if(attrScale.valType === "unique" ||
                    attrScale.valType === "nominal" ||
                    attrScale.valType === "nominal20" ||
                    attrScale.valType === "constant" || 
                    attrScale.valType === "constant-numeric") {
                    continue;
                } else if ($("#hid-filter").is(':checked') && attrScale.filterTypes.indexOf("hidden") !== -1) {
                    continue;
                } else if ($("#com-filter").is(':checked') && attrScale.filterTypes.indexOf("component") !== -1) {
                    continue;
                } else if ($("#rel-filter").is(':checked') && attrScale.filterTypes.indexOf("relation") !== -1) {
                    continue;
                }
            }
            xSelect.append(opt.clone());
            ySelect.append(opt.clone());
        }

        //populate the distance axis options
        for (var i = 1; i <= 4; i++){
            $('#anchor-'+i).html('<option value="none">None</option>');
        }

        for (var i = 0; i < clusters.length; i++) {
            var opt = $('<option>'+clusters[i].name+"</option>");
            opt.attr("value", clusters[i].name);
            for (var j = 1; j <= 4; j++){
                $('#anchor-'+j).append(opt.clone());
            }
        }

    }


    // A lot of this function is based on the Zoomable Circle Packing Example:
    // https://bl.ocks.org/mbostock/7607535
    ClusterRenderer.prototype.render = function() {
        var self = this;
        xScale.domain([0,local_clusters.length+1]).nice();
        yScale.domain([0,local_clusters.length+1]).nice();

        sizeScale.domain(d3.extent(local_clusters, function(d){
            return d.size;
        })).nice();

        vis.append('g')
            .attr('transform','translate(0,' + (h-padding*1.2) + ')')
            .attr('class','x axis')
            .attr('id','xaxis')
            .call(xAxis);

        vis.append('g')
            .attr('transform','translate('+(padding*1.2)+',0)')
            .attr('class','y axis')
            .attr('id','yaxis')
            .call(yAxis);

        var circles = vis.selectAll('circle')
            .data(local_clusters)
            .enter().append('circle')
            .attr('cx', function(d) { return xScale(d.index); })
            .attr('cy', function(d) { return yScale(d.index); })
            .attr('r', function(d) { return sizeScale(d.size); })
            .attr('class', function(d) {return 'cluster'; })
            .attr('id', function(d) { return d.name; })
            .on('click', function(d) { 
                if (self.focusNode !== d) {
                    updateAVTable(d); 
                    self.focusNode = d;
                    $("circle.focus").removeClass("focus");
                    $("#"+d.name).addClass("focus");
                    d3.event ? d3.event.stopPropagation() : undefined;
                }
            });

        vis.on("click", function() { 
            updateAVTable(local_data);
            self.focusNode = undefined;
            $("circle.focus").removeClass("focus");
        });

        vis.append("text")
            .attr("class","axis-label")
            .attr("text-anchor", "middle")  // this makes it easy to centre the text as the transform is applied to the anchor
            .attr("transform", "translate("+ (padding/2) +","+(h/2)+")rotate(-90)")  // text is drawn off the screen top left, move down and out and rotate
            .text("Cluster Index")
            .attr('id','left-label');

        vis.append("text")
            .attr("class","axis-label")
            .attr("text-anchor", "middle")  // this makes it easy to centre the text as the transform is applied to the anchor
            .attr("transform", "translate("+ (w/2) +","+(h-(padding/2))+")")  // centre below axis
            .text("Cluster Index")
            .attr('id','bottom-label');

        vis.append("text")
            .attr("class","axis-label")
            .attr("text-anchor", "middle")  // this makes it easy to centre the text as the transform is applied to the anchor
            .attr("transform", "translate("+ (w-(padding/2)) +","+(h/2)+")rotate(90)")  // text is drawn off the screen top left, move down and out and rotate
            .text("")
            .attr('id','right-label');

        vis.append("text")
            .attr("class","axis-label")
            .attr("text-anchor", "middle")  // this makes it easy to centre the text as the transform is applied to the anchor
            .attr("transform", "translate("+ (w/2) +","+(padding/2)+")")  // centre below axis
            .text("")
            .attr('id','top-label');
    }


    function updateAttrAxis() {
        if($('#cluster-style').val() !== 'attr'){ return; }
        var x_attr = $('#x-axis-attr').val();
        var y_attr = $('#y-axis-attr').val();

        // console.log(x_attr,y_attr);

        var new_x_scale = av_properties[x_attr];
        var new_y_scale = av_properties[y_attr];

        // console.log(new_x_scale.axisScale.domain(),new_y_scale.axisScale.domain());

        xScale.domain(new_x_scale.axisScale.domain());
        yScale.domain(new_y_scale.axisScale.domain());

        var t = vis.transition()
            .duration(750);

        t.selectAll("circle")
            .attr("cx", function(d) { 
                if(x_attr === "none"){ return xScale(d.index);}
                switch(new_x_scale.valType) {
                    case "binary":
                        return xScale(vt.binaryRatio(d,x_attr,local_data));
                    case "numeric":
                        return xScale(d.counts[x_attr][vt.CV_KEY].mean);
                }})
            .attr("cy", function(d) { 
                    if(y_attr === "none"){ return yScale(d.index);}
                    switch(new_y_scale.valType) {
                        case "binary":
                            return yScale(vt.binaryRatio(d,y_attr,local_data));
                        case "numeric":
                            return yScale(d.counts[y_attr][vt.CV_KEY].mean);
                    }});
        t.select("#xaxis").call(xAxis);
        t.select("#yaxis").call(yAxis);
        t.select("#bottom-label").text(x_attr === "none" ? "Cluster Index" : x_attr);
        t.select("#left-label").text(y_attr === "none" ? "Cluster Index" : y_attr);
        t.select("#top-label").text("");
        t.select("#right-label").text("");
    }

    function updateDistanceAxis() {
        if($('#cluster-style').val() !== 'dist'){ return; }

        var anchor1 = $("#anchor-1").val();
        var anchor2 = $("#anchor-2").val();

        // I'm currently assuming the distance is never > 1. I think that's true?
        xScale.domain([0,1]);
        yScale.domain([0,1]);

        var t = vis.transition()
            .duration(750);

        t.selectAll("circle") 
            .attr("cx", function(d) {
                if (anchor1 === "none") {
                    return xScale(0.5);
                }
                else if(anchor2 === "none") {
                    if(d.name === anchor1) {
                        return xScale(0.0);
                    }
                    else {
                        return xScale(distance_matrix[d.name][anchor1]);
                    }
                }
                else {
                    if (d.name === anchor1) { return xScale(0);}
                    else if (d.name === anchor2) { return xScale(1.0);}
                    else{
                        var a1 = distance_matrix[d.name][anchor1];
                        var a2 = distance_matrix[d.name][anchor2];
                        return xScale(a1 / (a1 + a2));
                    }
                }
            })
            .attr("cy", function(d) { 
                if (anchor1 !== "none" && anchor2 !== "none") {
                    if (d.name === anchor1 || d.name === anchor2) {
                        return yScale(0);
                    } 
                    else {
                        var a1 = distance_matrix[d.name][anchor1];
                        var a2 = distance_matrix[d.name][anchor2];
                        return yScale((a1+a2)/2);
                    }
                }
                else {
                    return yScale(0.5);
                }
            });

        t.select("#xaxis").call(xAxis);
        t.select("#yaxis").call(yAxis);
        if (anchor1 === "none"){    
            t.select("#top-label").text("");
            t.select("#left-label").text("");
            t.select("#bottom-label").text("");
            t.select("#right-label").text("");
        }
        else if(anchor2 == "none") {
            t.select("#top-label").text("");
            t.select("#left-label").text("Most Similar to "+anchor1);
            t.select("#bottom-label").text("");
            t.select("#right-label").text("Least Similar to "+anchor1);
        }
        else {         
            t.select("#top-label").text("Least Similar to "+anchor1+" and "+anchor2);
            t.select("#left-label").text("Most Similar to "+anchor1);
            t.select("#right-label").text("Most Similar to "+anchor2);
            t.select("#bottom-label").text("Most Similar to "+anchor1+" and "+anchor2);
        }
    }

    function updateSizeChanged() {
        console.log("updateSizeChanged");
        console.log($("#cluster-sizing").val());
        switch($("#cluster-sizing").val()) {
            case "abs":
                console.log("abs");
                sizeScale.domain(d3.extent(local_clusters, function(d){
                    return d.size;
                })).nice();

                var t = vis.transition()
                    .duration(750);
                t.selectAll("circle")
                    .attr('r', function(d){
                        return sizeScale(d.size);
                    });
                break;
            case "rel":
                console.log("rel");
                sizeScale.domain([0,local_data.size]);
                var t = vis.transition()
                    .duration(750);
                t.selectAll("circle")
                    .attr('r', function(d){
                        return sizeScale(d.size);
                    });
                break;
            case "equ":
                console.log("equ");
                var t = vis.transition()
                    .duration(750);
                t.selectAll("circle")
                    .attr('r', padding/2);
                break;
        }
    }


    function recolor(d,trgtAttr,scale) {
        // console.log(d)
        var circle = $("#" + d.name);
        // If the concept doesn't have the target attribute make it grey
        if (trgtAttr !== "none" && !d.counts[trgtAttr]) {
            circle.css("fill", defaultGrey);
            circle.css("fill-opacity",.5);
        } else {
            switch (scale.valType) {
                case "constant":
                    var val = Object.keys(d.counts[trgtAttr])[0];
                    circle.css("fill", scale.scale(d.counts[trgtAttr][val] / d.size));
                    circle.css("fill-opacity",.5);
                    break;

                case "constant-numeric":
                    circle.css("fill", scale.scale(d.counts[trgtAttr][vt.CV_KEY].n / d.size));
                    circle.css("fill-opacity",.5);
                    break;

                case "nominal":
                case "nominal20":
                    circle.css("fill", scale.scale(vt.majorityValue(d.counts,trgtAttr)));
                    circle.css("fill-opacity", d.counts[trgtAttr][vt.majorityValue(d.counts,trgtAttr)] / d.size);
                    break;

                case "binary":
                    if ($("#binary-style").val() === "categorical") {
                        circle.css("fill", scale.cat_scale(vt.majorityValue(d,trgtAttr)));
                        circle.css("fill-opacity", d.counts[trgtAttr][vt.majorityValue(d.counts,trgtAttr)] / d.size);
                    } else {
                        circle.css("fill", scale.num_scale(vt.binaryRatio(d,trgtAttr,local_data)));
                        circle.css("fill-opacity",.5);
                    }
                    break;

                case "numeric":
                    circle.css("fill", scale.scale(d.counts[trgtAttr][vt.CV_KEY].mean));
                    circle.css("fill-opacity",.5);
                    break;

                case "unique":
                default:
                    circle.css("fill", "#1f77b4");
                    circle.css("fill-opacity",.5);
                    break;
            }
        }
        if (d.children && d.children.length > 0) {
            for (var child in d.children) {
                recolor(d.children[child],trgtAttr,scale);
            }
        }
    }

    function populateColorByOptions() {
        var attrs = Object.keys(av_properties);
        attrs.sort();
        attrs.reverse();
        var colorBy = $("#focus-attr");
        var imgOptions = $("#img-attr");
        colorBy.html('<option value="none">None</option>');

        for (var i = 0; i < attrs.length; i++) {
            if (attrs[i] === "none") {continue;}
            var attrScale = av_properties[attrs[i]];

            var opt = $("<option>" + attrs[i] + "</option>");
            opt.attr("value", attrs[i]);
            imgOptions.append(opt.clone());

            if (attrScale) {
                if ($("#fil-val-unique").is(':checked') && attrScale.valType === "unique") {
                    continue;
                } else if ($("#fil-val-constant").is(':checked') && (attrScale.valType === "constant" || attrScale.valType === "constant-numeric")) {
                    continue;
                } else if ($("#hid-filter").is(':checked') && attrScale.filterTypes.indexOf("hidden") !== -1) {
                    continue;
                } else if ($("#com-filter").is(':checked') && attrScale.filterTypes.indexOf("component") !== -1) {
                    continue;
                } else if ($("#rel-filter").is(':checked') && attrScale.filterTypes.indexOf("relation") !== -1) {
                    continue;
                }
            }
            colorBy.append(opt);
        }
        colorBy.change(this.colorSelectChanged);
    }

    ClusterRenderer.prototype.colorSelectChanged = function() {
        var attr = $("#focus-attr").val();
        if (attr === "none") {
            $("#focus-attr-type").text("Default");
            $("#focus-val-type").text("Default");
            $(".color-ops").hide();
        } else {
            var attrScale = av_properties[attr];
            switch (attrScale.valType) {
                case "unique":
                    $("#focus-val-type").text("Unique");
                    $("#numeric-colors").hide();
                    $("#nominal-colors").hide();
                    break;
                case "constant":
                case "constant-numeric":
                    $("#focus-val-type").text("Constant");
                    $("#numeric-colors").hide();
                    $("#nominal-colors").hide();
                    break;
                case "nominal":
                case "nominal20":
                    $("#focus-val-type").text("Nominal");
                    var colorTable = $("#nom-val-colors");
                    colorTable.html('');
                    var vals = Object.keys(local_data.counts[attr]);
                    vals.sort();
                    vals.reverse();
                    for (i = 0; i < vals.length; i++) {
                        var tr = $("<tr></tr>");
                        var valName = $("<td>" + vals[i] + "</td>");
                        var colorCell = $("<td></td>");
                        valName.addClass("absorbing-column");
                        colorCell.addClass("swatch");
                        colorCell.css("background-color", attrScale.scale(vals[i]));
                        tr.append(valName);
                        tr.append(colorCell);
                        colorTable.append(tr);
                    }
                    $("#numeric-colors").hide();
                    $("#nominal-colors").show();
                    break;
                case "numeric":
                    $("#focus-val-type").text("Numeric");
                    var dom = attrScale.scale.domain();
                    $("#num-val-1").text(dom[0].toFixed(3));
                    $("#num-val-9").text(dom[1].toFixed(3));
                    $("#num-val-5").text(((dom[1] + dom[0]) / 2).toFixed(3));
                    for (var i = 0; i < 10; i++) {
                        $("#num-col-" + i).css("background-color", attrScale.scale(dom[0] + (dom[1] - dom[0]) / 9 * i));
                    }
                    $("#numeric-colors").show();
                    $("#nominal-colors").hide();
                    break;
                case "binary":
                    $("#focus-val-type").text("Binary");
                    var rootVals = Object.keys(data.counts[attr]);
                    rootVals.sort();
                    rootVals.reverse();
                    if (isPositiveBinary(rootVals)) {
                        rootVals.reverse();
                    }
                    if ($("#binary-style").val() === "categorical") {
                        var colorTable = $("#nom-val-colors");
                        colorTable.html('');
                        for (i = 0; i < rootVals.length; i++) {
                            var tr = $("<tr></tr>");
                            var valName = $("<td>" + rootVals[i] + "</td>");
                            var colorCell = $("<td></td>");
                            valName.addClass("absorbing-column");
                            colorCell.addClass("swatch");
                            colorCell.css("background-color", attrScale.cat_scale(rootVals[i]));
                            tr.append(valName);
                            tr.append(colorCell);
                            colorTable.append(tr);
                        }
                        $("#numeric-colors").hide();
                        $("#nominal-colors").show();
                    } else {
                        $("#num-val-1").text(rootVals[0]);
                        $("#num-val-9").text(rootVals[1]);
                        $("#num-val-5").text("");
                        for (var i = 0; i < 10; i++) {
                            $("#num-col-" + i).css("background-color", attrScale.num_scale(i / 9));
                        }
                        $("#numeric-colors").show();
                        $("#nominal-colors").hide();
                    }
                    break;
                default:
                    console.error("unknown attribute type found");
                    console.log(local_data);
            }
            switch (attrScale.attrType) {
                case "hidden-relation":
                    $("#focus-attr-type").text("Hidden Relation");
                    break;
                case "component-relation":
                    $("#focus-attr-type").text("Component Relation");
                    break;
                case "constant-relation":
                    $("#focus-attr-type").text("Constant Relation");
                    break;
                case "variable":
                    $("#focus-attr-type").text("Variable");
                    break;
                case "hidden":
                    $("#focus-attr-type").text("Hidden");
                    break;
                case "constant":
                default:
                    $("#focus-attr-type").text("Constant");
                    break;
            }
        }
        recolor(local_data,attr,av_properties[attr]);
    }

    function updateAVTable(node_data) {
        d = node_data.counts

        var property_sheet = $("#properties");
        var prop_filter = $("#property-filter");
        property_sheet.html('');

        $("#concept-name").text(node_data['name'])
        $("#concept-size").text(node_data['size'])
        var n_children = node_data['children'] ? node_data['children'].length : 0;
        $("#concept-child-size").text(n_children)
        var d_keys = Object.keys(d);
        var attrs = []

        for (var i = 0; i < d_keys.length; i++) {
            var attrScale = av_properties[d_keys[i]];

            if (attrScale) {
                if ($("#fil-val-unique").is(':checked') && attrScale.valType === "unique") {
                    continue;
                }
                if ($("#fil-val-constant").is(':checked') && (attrScale.valType === "constant" || attrScale.valType === "constant-numeric")) {
                    continue;
                }
                if ($("#fil-val-binary").is(':checked') && attrScale.valType === "binary") {
                    continue;
                }
                if ($("#fil-val-numeric").is(':checked') && attrScale.valType === "numeric") {
                    continue;
                }
                if ($("#fil-val-nominal").is(':checked') && (attrScale.valType === "nominal" || attrScale.valType === "nominal20")) {
                    continue;
                }
                if ($("#hid-filter").is(':checked') && attrScale.filterTypes.indexOf("hidden") !== -1) {
                    continue;
                }
                if ($("#com-filter").is(':checked') && attrScale.filterTypes.indexOf("component") !== -1) {
                    continue;
                }
                if ($("#rel-filter").is(':checked') && attrScale.filterTypes.indexOf("relation") !== -1) {
                    continue;
                }
                attrs.push(d_keys[i]);
            }

        }
        attrs.sort();
        attrs.reverse();
        for (var a in attrs) {
            attr = attrs[a]
            var tr = $("<tr></tr>");
            var ar = $('<td colspan="3">' + attr + '</td>');
            ar.addClass("prop-column");
            tr.addClass("info");
            tr.append(ar);
            property_sheet.append(tr);
            vals = Object.keys(d[attr]);
            vals.sort();
            vals.reverse();
            for (var v in vals) {
                val = vals[v];
                tr = $("<tr></tr>");
                var vd = $("<td></td>");
                var n = NaN;
                //if this is a continuous value
                if (val === vt.CV_KEY) {
                    vd.text(d[attr][val].mean.toFixed(3) + " (" + d[attr][val].std.toFixed(3) + ")");
                    n = d[attr][val].n;
                }
                //else
                else {
                    vd.text(val);
                    n = d[attr][val];
                }
                vd.addClass("val-row");
                tr.append(vd);
                var cd = $("<td>" + n + "</td>");
                var pd = $("<td>" + (n / node_data['size'] * 100).toFixed(2) + "%</td>")
                tr.append(cd);
                tr.append(pd);
                property_sheet.append(tr);
            }
        }
    }

    ClusterRenderer.prototype.updateFilters = function() {
        populateColorByOptions();
        this.colorSelectChanged();
        updateAVTable(local_data);
    }

    ClusterRenderer.prototype.searchConceptByName = function(e) {
        var conc_name = $("#search-concept-name").val();
        if (conc_name.length === 0) {
            $(".focus").removeClass("focus");
            $("#search-not-found").hide();
            return;
        }

        if (!conc_name.startsWith("Concept")) {
            conc_name = "Concept" + conc_name;
        }

        if (/Concept\d+/g.test(conc_name)) {
            if ($("#" + conc_name).length === 0) {
                $("#search-not-found").text("No " + conc_name);
                $("#search-not-found").show();
            } else {
                $("#search-not-found").hide();
                $(".focus").removeClass("focus");
                var n = undefined;
                for (var i = nodes.length - 1; i >= 0; i--) {
                    if (nodes[i].data.name == conc_name) {
                        n = nodes[i];
                    }
                }
                if (n !== undefined) g.select("#" + conc_name).on("click")(n);
                $("#" + conc_name).addClass("focus");
            }
        }
    }
    return ClusterRenderer;
})();

