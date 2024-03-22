conventions using examples

```
struct Foo { /* */ }

// optional
fn foo_new(/* */) -> Foo { /* */ }

// private fn
fn _foo_do_something_idk(foo: Foo, /* */) -> ? { /* */ }

//public fn
fn foo_do_something_else(foo: Foo, /* */) -> ? { /* */ }

fn bar() {
    let asdad = Foo(
        /* field name 1 */ some_value,
        /* field name 2 */ some_value,
        /* field name 3 */ some_value,
    );
}
```

so basically, a "member" function to some struct must start with the struct's name in lower_snake_case
if said function is "private", it should be prefixed by an underscore

when initialising a struct, one must list the field names in /* */ comments befor ethe values

functions "private" to specific files must be prefixed with "_file_name"
this might cause conflicts with private member functions but oh well

no function name may start with an underscore and a capital letter as its prefix
