const cv_key = "#ContinuousValue#"; 

      var data = {}

      var w = 1280,
        h = window.innerHeight,
        r = 720,
        x = d3.scale.linear().range([0, r]),
        y = d3.scale.linear().range([0, r]),
        node,
        root;

      var pack = d3.layout.pack()
                  .size([r, r])
                  .value(function(d) { return d.size; });

      var vis = d3.select("div#plot-panel")
            .append("div")
            .classed("svg-container",true)
            .insert("svg:svg", "h2")
            .attr("preservAspectRatio", "xMinYMin meet")
            .attr("viewBox", "0 0 750 750")
            .classed("svg-content-responsive", true);
            /* .attr("transform", "translate(" + (w - r) / 2 + "," + (h - r) / 2 + ")"); */
            /* in Firefox does not center the circle in the panel correctly */

      var attributeScales = {};
      var redGreenScale = d3_scale.scaleLinear()
                              .domain([0,1])
                              .range(["Red","LimeGreen"]);
      var saturationScale = d3_scale.scaleLinear()
                              .domain([0,1])
                              .range(["#cccccc","steelblue"]);
      var defaultGrey = "#cccccc";

      function parseNumeric(numeric) {
        var patt = /\d+\.\d+\s\(\d+\.\d+\)\s\[\d+\]/g;
        if (patt.test(numeric)) {
          var nm = numeric.split(" ");
          var mean = parseFloat(nm[0]);
          var std = parseFloat(nm[1].substring(1,nm[1].length-1));
          var count = parseFloat(nm[2].substring(1,nm[1].length-1));
          return {"mean":mean,"std":std,"min":mean-2*std,"max":mean+2*std,"count":count}
        }
        else {
          return null;
        }
      }

      function augmentNumeric(numeric){
        numeric["min"] = numeric["mean"] - 2 * numeric["std"];
        numeric["max"] = numeric["mean"] + 2 * numeric["std"];
        return numeric;
      }

      /*
       * unary - there is only 1 value and it is not numeric
       * numeric - there is only 1 value and it is a distribution
       * binary - there are only 2 values
       * nominal - there are more than 2 values
       * nominal20 - there are more than 20 values
       * unique - n(values) === n(instances) 
       *
       */
      function attributeType(rootObj,attr){
        var vals = Object.keys(rootObj['counts'][attr]);

        if(vals.length === rootObj['size']) {
          return 'unique';
        }
        if(vals.length > 10) {
          return 'nominal20';
        }
        if(vals.length > 2) {
          return 'nominal';
        }
        if(vals.length > 1) {
          return 'binary';
        }
        // example - '0.0000 (0.0000) [2]'
        //var patt = /\d+\.\d+\s\(\d+\.\d+\)\s\[\d+\]/g;
        //var patt = /#ContinuousValue#/g;
        if(vals[0] === cv_key) {
          return "numeric";
        }
        else {
          return "unary";
        }
      }

      function isRelation(val) {
        var patt = /\('.+'\,.*\)/g;
        return patt.test(val);
      }

      function filterTypes(attr) {
        var typs = []
        if(attr[0] === '_'){
          typs.push('hidden');
        }
        if(/\('.+'\,.*\)/g.test(attr)){
          typs.push('relation');
        }
        if(/\('.+'\,'\?.+'\)/g.test(attr)) {
          typs.push('component');
        }
        return typs;
      }   

      function setupAttributes(rootObj) {
        attributeScales["none"] = {"type":"default","scale":null};
        var attrs = Object.keys(rootObj['counts']);
        for (var i = 0; i < attrs.length; i ++) {
          var attr = attrs[i];
          var vals = Object.keys(rootObj['counts'][attr]);
          vals.sort();
          vals.reverse();
          var attType = attributeType(rootObj,attr);
          //var filType = filterType(rootObj,attr);
          var at_scale = {"type":attType,"scale":null,"filterTypes":filterTypes(attr)};
          if (attType === "nominal") {
            at_scale.scale = d3_scale.scaleCategory10();
          }
          else if (attType === "nominal20") {
            at_scale.scale = d3_scale.scaleCategory20();
          }
          else if (attType === "binary") {
            at_scale.scale = redGreenScale; //d3_scale.scalePlasma();
          }
          else if (attType === "numeric") {
            //var num = parseNumeric(rootObj['counts'][attr]['#ContinuousValue#']);
            var num = augmentNumeric(rootObj['counts'][attr][cv_key]);
            at_scale.scale = redGreenScale.copy().domain([num.min,num.max]);
          }
          else if (attType === "unary") {
            at_scale.scale = saturationScale;
          }
          attributeScales[attr] = at_scale;
        }
      }

      function randomImage(d,attr) {
        if(attr === "none"){
            return "";
        }
        if(d.counts){
            d=d.counts;
        }
        var vals = Object.keys(d[attr]);
        return  "images/" + vals[Math.floor(Math.random() * vals.length)] + ".jpg";
      }

      function hasImageData(d) {
        if(d.counts){
            d=d.counts;
        }
        return d[$("#img-attr").val()];
      }

      function imageSelectChanged() {
        changeImages(data);
      }

      function changeImages(d){
        var img = $("#"+d.name+"-img");
        if(img){
            img.attr("href",randomImage(d,$("#img-attr").val()));
        }
        if(d.children) {
            for(var i in d.children){
              changeImages(d.children[i]);
            }
        }
      }

      function changeText() {
        $("text").css("display", $("#show-names").is(':checked') ? "block" : "none");
      }

      //This is mainly built on the d3 circle packing example
      function buildTree(tree_data) {
        node = root = data = tree_data;

        var nodes = pack.nodes(root);

        vis.selectAll("circle").remove();
        vis.selectAll("image").remove();
        vis.selectAll("text").remove();

        vis.selectAll("circle")
        .data(nodes)
          .enter().append("svg:circle")
            .attr("id", function(d) { return d.name; })
            .attr("class", function(d) { return d.children ? "parent" : "child"; })
            .attr("cx", function(d) { return d.x; })
            .attr("cy", function(d) { return d.y; })
            .attr("r", function(d) { return d.r; })
            .on("click", function(d) { return zoom(node == d ? root : d); });
            //.style("opacity", function(d) { return (d.r > 100) ? 1 : 0; });

          vis.selectAll("image.screenshot")
          .data(nodes.filter(function(d){
                  if (d.children){// || !hasImageData(d)){ 
                    return false; 
                  } else {
                    return true;
                  }}))
            .enter().append("image")
                .attr("id",function(d) { return d.name+"-img";})
                .attr('class', 'screenshot')
                .attr("xlink:href", function(d) { return randomImage(d,$("#img-attr").val());})
            .attr("width", function(d) { return (1.33 * d.r); })
            .attr("height", function(d) { return d.r; })
            .attr("x", function(d) { return d.x - ((1.33 * d.r)/2); })
            .attr("y", function(d) { return d.y - (d.r/2); })
            .on("click", function(d) { return zoom(node == d ? root : d); });

        vis.selectAll("text")
            .data(nodes)
            .enter().append("svg:text")
            .attr("class", function(d) { return d.children ? "parent" : "child"; })
            .attr("x", function(d) { return d.x; })
            .attr("y", function(d) { return d.y; })
            .attr("dy", ".35em")
            .attr("text-anchor", "middle")
            .text(function(d) { return d.name; })
            .on("click", function(d) { return zoom(node == d ? root : d); });

        //d3.select(window).on("click", function() { zoom(root); });
        vis.on("click", function() { zoom(root); });
        changeText();
      }

      function zoom(d, i) {
        //console.log(d);
        make_property_sheet(d);

        scale = 1.0;
        //console.log(d.CU)
        //console.log(d.counts)
        var k = r / d.r / 2;
        x.domain([d.x - d.r, d.x + d.r]);
        y.domain([d.y - d.r, d.y + d.r]);

        var t = vis.transition()
            .duration(d3.event.altKey ? 7500 : 750);

        t.selectAll("image.screenshot")
            .attr("width", function(d) { return (scale * 1.33 * k * d.r); })
            .attr("height", function(d) { return scale * k * d.r; })
            .attr("x", function(d) { return scale * (x(d.x) - ((1.33 * k * d.r)/2)); })
            .attr("y", function(d) { return scale * (y(d.y) - ((k * d.r)/2)); });

        t.selectAll("circle")
            .attr("cx", function(d) { return scale * x(d.x); })
            .attr("cy", function(d) { return scale * y(d.y); })
            .attr("r", function(d) { return scale * k * d.r; });
            //.style("opacity", function(d) { return (k * d.r > 100) ? 1 : 0; });

        t.selectAll("text")
            .attr("x", function(d) { return scale * x(d.x); })
            .attr("y", function(d) { return scale * y(d.y); })
            //.style("opacity", function(d) { return (k * d.r > 100 && k * d.r < 200) ? 1 : 0; });

        node = d;
        d3.event.stopPropagation();
      }

      function majorityValue(table) {
        var values = Object.keys(table);
        values.sort();
        values.reverse();
        var maj = {"n":"","v":0};
        for (var i = 0; i < values.length; i++) {
          if(table[values[i]] > maj.v){
            maj.n = values[i];
            maj.v = table[values[i]];
          }
        }
        return maj.n;
      }

      // a bit of a hack to make sure binary variaables that have sensible meaning are ordered correctly
      // the values in each list are based on the test datasets we have and my own thoughts
      function isPositiveBinary(vals) {
        return $.inArray(vals[0].toLowerCase(), ["success","yea","yes","y","pro","pass"]) > -1 &&
              $.inArray(vals[1].toLowerCase(), ["failure", "fail","nay","no","n","con"]) > -1;
      }

      function binaryRatio(table,targetAttr) {
        var rootVals = Object.keys(data['counts'][targetAttr]);
        rootVals.sort();
        rootVals.reverse();
        var values = Object.keys(table);
        values.sort();
        values.reverse();

        //Special case to properly sort a success/failure binary
        if(isPositiveBinary(rootVals)){
          values.reverse();
          rootVals.reverse();
        }

        if(values.length === 1) {
          if (values[0] === rootVals[0]){
            return 0.0;
          }
          else {
            return 1.0;
          }
        }
        else {
          return table[values[1]]/(table[values[0]]+table[values[1]]);
        }
      }

      function recolor(d) {
        var targetAttr = $("#color-by").val();
        var scale = attributeScales[targetAttr];
        if(!scale){
          console.log("No scale found for attribute:" + targetAttr);
          return;
        }
        var circle = $("#"+d.name);
        if (targetAttr !== "none" && !d['counts'][targetAttr]) {
          circle.css("fill",defaultGrey);
        }
        else {
          switch(scale.type) {
            case "unary":
              var val = Object.keys(d['counts'][targetAttr])[0];
              circle.css("fill",scale.scale(d['counts'][targetAttr][val]/d['size']));
              break;
            
            case "nominal":
            case "nominal20":
              circle.css("fill",scale.scale(majorityValue(d['counts'][targetAttr])));
              break;
            
            case "binary":
              circle.css("fill",scale.scale(binaryRatio(d['counts'][targetAttr],targetAttr)));
              break;
            
            case "numeric":
              //var num = parseNumeric(Object.keys(d['counts'][targetAttr])[0]);
              //var num = augmentNumeric(Object.keys(d['counts'][targetAttr][cv_key]))
              var num = d['counts'][targetAttr][cv_key];
              circle.css("fill",scale.scale(num.mean));
              break;
            
            case "unique":
            default:
              if(d.children) {
                circle.css("fill","steelblue");
              }
              else {
                circle.css("fill",defaultGrey);
              }
              break;
          }
        }
        if(d.children){
            for(var child in d.children){
              recolor(d.children[child]);
            }
        }

      }

      function populateColorByOptions(d) {
        var attrs = Object.keys(d['counts']);
        attrs.sort();
        attrs.reverse();
        var colorBy = $("#color-by");
        var imgOptions = $("#img-attr");
        colorBy.html('<option value="none">None</option>');
        var hid_filter = $("#hid-filter").is(':checked');
        var rel_filter = $("#rel-filter").is(':checked');
        var una_filter = $("#una-filter").is(':checked');
        var unq_filter = $("#unq-filter").is(':checked');
        var com_filter = $("#com-filter").is(':checked');

        for (var i=0; i < attrs.length;i++){
          var attrScale = attributeScales[attrs[i]];
          
          var opt = $("<option>"+attrs[i]+"</option>");
          opt.attr("value",attrs[i]);
          imgOptions.append(opt.clone());

          if(attrScale) {
            if(unq_filter && attrScale.type === "unique") {
              continue;
            }
            else if(una_filter && attrScale.type === "unary") {
              continue;
            }
            else if(hid_filter && attrScale.filterTypes.indexOf("hidden") !== -1){
              continue;
            }
            else if(com_filter && attrScale.filterTypes.indexOf("component") !== -1) {
              continue;
            }
            else if(rel_filter && attrScale.filterTypes.indexOf("relation") !== -1) {
              continue;
            }
          }
          colorBy.append(opt);
        }
        colorBy.change(colorSelectChanged);
      }

      function colorSelectChanged() {
        var attr = $("#color-by").val();
        if(attr === "none"){
          $("#color-by-type").text("Default");
          $(".color-ops").hide();
        }
        else {
          var attrScale = attributeScales[attr];
          if(attrScale.type === "unique") {
            $("#color-by-type").text("Unique");
          }
          else if(attrScale.type === "unary") {
            $("#color-by-type").text("Unary"); 
          }
          if(attrScale.type ==="nominal" || attrScale.type === "nominal20"){
            $("#color-by-type").text("Nominal");
            var colorTable = $("#nom-val-colors");
            colorTable.html('');
            var vals = Object.keys(data['counts'][attr]);
            vals.sort();
            vals.reverse();
            for(i=0;i<vals.length;i++) {
              var tr = $("<tr></tr>");
              var valName = $("<td>"+ vals[i] +"</td>");
              var colorCell = $("<td></td>");
              valName.addClass("absorbing-column");
              colorCell.addClass("swatch");
              colorCell.css("background-color",attrScale.scale(vals[i]));
              tr.append(valName);
              tr.append(colorCell);
              //tr.append('<td class="absorbing-column"></td>');
              colorTable.append(tr);
            }
            $("#numeric-colors").hide();
            $("#nominal-colors").show();
          }
          else if (attrScale.type ==="numeric"){
            $("#color-by-type").text("Numeric");
            $("#col-min-val").text(attrScale.scale.domain()[0].toFixed(3));
            $("#col-min-color").css("background-color",attrScale.scale.range()[0]);
            $("#col-max-val").text(attrScale.scale.domain()[1].toFixed(3));
            $("#col-max-color").css("background-color",attrScale.scale.range()[1]);
            $("#numeric-colors").show();
            $("#nominal-colors").hide();
          }
          else if (attrScale.type === "binary") {
            $("#color-by-type").text("Binary");
            var rootVals = Object.keys(data['counts'][attr]);
            rootVals.sort();
            rootVals.reverse();
            //Special case to properly sort a success/failure binary
            if(isPositiveBinary(rootVals)){
              rootVals.reverse();
            }
            
            $("#col-min-val").text(rootVals[0]);
            $("#col-min-color").css("background-color",attrScale.scale.range()[0]);
            $("#col-max-val").text(rootVals[1]);
            $("#col-max-color").css("background-color",attrScale.scale.range()[1]);
            $("#numeric-colors").show();
            $("#nominal-colors").hide();
          }
          else {
            console.error("unknown attribute type found");
            console.log(data);
          }
        }
        recolor(data);
      }

      function make_property_sheet(node_data) {
        d = node_data['counts']
        var property_sheet = $("#properties"); 
        var prop_filter = $("#property-filter");
        property_sheet.html('');

        $("#concept-name").text(node_data['name'])
        $("#concept-size").text(node_data['size'])
        var n_children = node_data['children'] ? node_data['children'].length : 0;
        $("#concept-child-size").text(n_children)

        var d_keys = Object.keys(d);
        var attrs = []
        var hid_filter = $("#hid-filter").is(':checked');
        var una_filter = $("#una-filter").is(':checked');
        var rel_filter = $("#rel-filter").is(':checked');
        var unq_filter = $("#unq-filter").is(':checked');
        var com_filter = $("#com-filter").is(':checked');

        for (var i = 0; i < d_keys.length; i++){
          var attrScale = attributeScales[d_keys[i]];

          if(attrScale) {
            if(unq_filter && attrScale.type === "unique") {
              continue;
            }
            else if(una_filter && attrScale.type === "unary") {
              continue;
            }
            else if(hid_filter && attrScale.filterTypes.indexOf("hidden") !== -1){
              continue;
            }
            else if(com_filter && attrScale.filterTypes.indexOf("component") !== -1) {
              continue;
            }
            else if(rel_filter && attrScale.filterTypes.indexOf("relation") !== -1) {
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
          var ar = $("<td>"+attr+"</td>");
          tr.addClass("info");
          tr.append(ar);
          tr.append("<td></td><td></td>");
          property_sheet.append(tr);
          vals = Object.keys(d[attr]);
          vals.sort();
          vals.reverse();
          for (var v in vals){
            val = vals[v];
            tr = $("<tr></tr>");
            var vd = $("<td></td>");
            var n = NaN;
            //if this is a continuous value
            if(val === cv_key){
              vd.text(d[attr][val].mean.toFixed(3) + " (" + d[attr][val].std.toFixed(3)+ ")");
              n = d[attr][val].n;
            }
            //else
            else {
              vd.text(val);
              n = d[attr][val];
            }
            vd.addClass("val-row");
            tr.append(vd);
            var cd = $("<td>"+n+"</td>");
            var pd = $("<td>"+(n/node_data['size']*100).toFixed(2)+"%</td>")
            tr.append(cd);
            tr.append(pd);
            property_sheet.append(tr);
          }
        }
      }

      function updateFilters() {
        populateColorByOptions(data);
        colorSelectChanged();
        make_property_sheet(node);
      }

      
