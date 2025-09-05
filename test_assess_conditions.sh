#!/bin/bash

# Test the assess-conditions endpoint with proper JSON formatting
curl -X POST http://localhost:8000/assess-conditions \
  -H "Content-Type: application/json" \
  -d '{
    "search_query": "bicycle in san diego",
    "results": [
      {
        "id": "1417273589571573",
        "title": "Conversion kit bike ( 🚨GOES 38 MPH 🚨 )",
        "price": "$1",
        "source": "facebook",
        "image": "https://scontent-lax3-1.xx.fbcdn.net/v/t45.5328-4/541671374_1430551655093177_7488556763206710628_n.jpg?stp=c0.43.261.261a_dst-jpg_p261x260_tt6&_nc_cat=104&ccb=1-7&_nc_sid=247b10&_nc_ohc=5JRII2o-Bq8Q7kNvwEAGnPn&_nc_oc=Adn4M-MUCbXbdBmVdY0VEItyJpVHMdosWwfhElZ6o5KCpt2Yorex4XOXp0HRw1mS9ppGDjZzweM2lN8fMnhgxbTC&_nc_zt=23&_nc_ht=scontent-lax3-1.xx&_nc_gid=vKzan1MBsubXvyKJss7JwQ&oh=00_AfYJ-Sj6HnfpN5ybiRUHvOZr_4n7RIwSGTRmXGQfmw5KnA&oe=68BD75E1",
        "location": "Moreno Valley, CA",
        "url": "https://www.facebook.com/marketplace/item/1417273589571573"
      },
      {
        "id": "726707020531733",
        "title": "Beach cruiser - Pink/white - Adult",
        "price": "$10",
        "source": "facebook",
        "image": "https://scontent-lax3-2.xx.fbcdn.net/v/t45.5328-4/532113155_1313665073660869_8714300154800246823_n.jpg?stp=c0.151.261.261a_dst-jpg_p261x260_tt6&_nc_cat=100&ccb=1-7&_nc_sid=247b10&_nc_ohc=PuKCuYUq7F4Q7kNvwHNkJo_&_nc_oc=Adlv1B0jSKQRC1M7wsqjY-zrEl2bdw2BeeBt_5Ui4W-H247oA8ZTpDRjB5xExi2gWU4JNP7vc6r8ExuqRY8eW3lM&_nc_zt=23&_nc_ht=scontent-lax3-2.xx&_nc_gid=8-DEcySt5NX0uzg96HWUyg&oh=00_AfaXXHDxw7sLf-Q-m7bR0uGF1LwH9PS_xqqhFhsacNxCLA&oe=68BD869B",
        "location": "San Diego, CA",
        "url": "https://www.facebook.com/marketplace/item/726707020531733"
      },
      {
        "id": "1632143021075844",
        "title": "Boys bike",
        "price": "$20",
        "source": "facebook",
        "image": "https://scontent-lax3-1.xx.fbcdn.net/v/t45.5328-4/541627567_1287676782959964_7397613618869335601_n.jpg?stp=c151.0.260.260a_dst-jpg_p261x260_tt6&_nc_cat=104&ccb=1-7&_nc_sid=247b10&_nc_ohc=klAYQ6q87LkQ7kNvwFW_jL1&_nc_oc=AdnuTnnFhT2yeifBsNKMpXTSYY-nKoTxk1Mk5qRL5Ol9Jx07QY3HfHyv3mzaI9IoArZdNvDGEglMuxOIwOqZ6zgW&_nc_zt=23&_nc_ht=scontent-lax3-1.xx&_nc_gid=vKzan1MBsubXvyKJss7JwQ&oh=00_AfYWUXwvGC2tqrh34S2Lc9R0skvMq5gvLMse5oSb6jkT_w&oe=68BD606C",
        "location": "San Diego, CA",
        "url": "https://www.facebook.com/marketplace/item/1632143021075844"
      },
      {
        "id": "1496154951826849",
        "title": "Huffy",
        "price": "$20",
        "source": "facebook",
        "image": "https://scontent-lax3-2.xx.fbcdn.net/v/t45.5328-4/531983031_2297615570695827_3348399317273197742_n.jpg?stp=c43.0.260.260a_dst-jpg_p261x260_tt6&_nc_cat=111&ccb=1-7&_nc_sid=247b10&_nc_ohc=nYyWw9kN_gwQ7kNvwEkaDz1&_nc_oc=AdlLsdl-nmXS0hM1BTa6z_rPHxDMxs-S0_kmFHZoxTeU6DK8VcbPIsGysIGV-YmhN6DolX4UoyyN-Jep3aB1d6B8&_nc_zt=23&_nc_ht=scontent-lax3-2.xx&_nc_gid=8-DEcySt5NX0uzg96HWUyg&oh=00_AfZ9Bi-nIHpKBJope1eSjAJCPONCAhvFls9abTlXj9edJw&oe=68BD9405",
        "location": "San Marcos, CA",
        "url": "https://www.facebook.com/marketplace/item/1496154951826849"
      },
      {
        "id": "1611026772911562",
        "title": "Bike",
        "price": "$30",
        "source": "facebook",
        "image": "https://scontent-lax3-2.xx.fbcdn.net/v/t45.5328-4/498012295_686722517396197_1149402998871777852_n.jpg?stp=c0.152.261.261a_dst-jpg_p261x260_tt6&_nc_cat=107&ccb=1-7&_nc_sid=247b10&_nc_ohc=R7WTXQZX39QQ7kNvwF4K2hc&_nc_oc=AdlPDaApjIaZC-PqERJV6kjSvZS6Wp9z6MUIlVGvS-e9gyB-axESLfa3qqzInJsYroyYGeSNn-eH5dtQMcX3DaSA&_nc_zt=23&_nc_ht=scontent-lax3-2.xx&_nc_gid=KSTfipGwvJMTFWSPcAKvag&oh=00_AfbmQ5muHyXW5rCp7MMvZmmLoab7Lz8k_ZWwtcUNlqA2dQ&oe=68BD6521",
        "location": "Escondido, CA",
        "url": "https://www.facebook.com/marketplace/item/1611026772911562"
      }
    ]
  }'
