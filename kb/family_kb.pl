% --- Facts (14) ---
parent(john, mary).
parent(john, paul).
parent(susan, mary).
parent(susan, paul).

parent(mary, alice).
parent(mary, bob).
parent(dave, alice).
parent(dave, bob).

parent(paul, emma).
parent(lisa, emma).

male(john).
male(paul).
male(dave).
male(bob).

female(susan).
female(mary).
female(lisa).
female(alice).
female(emma).

% --- Rules (9) ---
father(X,Y) :- parent(X,Y), male(X).
mother(X,Y) :- parent(X,Y), female(X).

sibling(X,Y) :- parent(P,X), parent(P,Y), neq(X,Y).

grandparent(X,Z) :- parent(X,Y), parent(Y,Z).

ancestor(X,Y) :- parent(X,Y).
ancestor(X,Y) :- parent(X,Z), ancestor(Z,Y).

aunt(X,Y) :- female(X), sibling(X,P), parent(P,Y).
uncle(X,Y) :- male(X), sibling(X,P), parent(P,Y).

% inequality helper (no built-in \= in our tiny engine)
neq(a,b). neq(a,c). neq(a,d). neq(a,e). neq(a,f). neq(a,g). neq(a,h).
neq(b,a). neq(b,c). neq(b,d). neq(b,e). neq(b,f). neq(b,g). neq(b,h).
neq(c,a). neq(c,b). neq(c,d). neq(c,e). neq(c,f). neq(c,g). neq(c,h).
neq(d,a). neq(d,b). neq(d,c). neq(d,e). neq(d,f). neq(d,g). neq(d,h).
neq(e,a). neq(e,b). neq(e,c). neq(e,d). neq(e,f). neq(e,g). neq(e,h).
neq(f,a). neq(f,b). neq(f,c). neq(f,d). neq(f,e). neq(f,g). neq(f,h).
neq(g,a). neq(g,b). neq(g,c). neq(g,d). neq(g,e). neq(g,f). neq(g,h).
neq(h,a). neq(h,b). neq(h,c). neq(h,d). neq(h,e). neq(h,f). neq(h,g).
